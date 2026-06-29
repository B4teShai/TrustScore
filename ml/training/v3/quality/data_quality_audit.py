"""Phase 2 — data quality + bias audit across v3-relevant datasets.

Reports missing values, exact + near-duplicate rates, class imbalance, text-length
distribution shift between domains (KS), and a cheap label-noise probe. Writes
data_quality_report.{json,md} and bias_report.md. Leakage is handled separately in
leakage_analysis.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from ml.training.common import clean_text, write_json
    from ml.training.v3 import DATA_DIR, REPORTS_DIR, SEED
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[3]))
    from training.common import clean_text, write_json  # type: ignore
    from training.v3 import DATA_DIR, REPORTS_DIR, SEED  # type: ignore

_WS = re.compile(r"\s+")


def _norm_hash(text: str) -> str:
    return hashlib.md5(_WS.sub(" ", str(text).lower().strip()).encode()).hexdigest()


def _audit_text_dataset(path: Path, label_col: str | None, cap: int) -> dict[str, Any]:
    df = pd.read_csv(path, low_memory=False)
    if len(df) > cap:
        df = df.sample(n=cap, random_state=SEED).reset_index(drop=True)
    text_col = "text" if "text" in df.columns else df.columns[0]
    txt = df[text_col].fillna("").astype(str)
    norm = txt.map(_norm_hash)
    out: dict[str, Any] = {
        "rows_sampled": int(len(df)),
        "missing_values": {c: int(df[c].isna().sum()) for c in df.columns},
        "empty_text_rate": round(float((txt.str.len() == 0).mean()), 4),
        "exact_duplicate_rate": round(float(txt.duplicated().mean()), 4),
        "near_duplicate_rate": round(float(norm.duplicated().mean()), 4),
        "text_len_chars": {"mean": round(float(txt.str.len().mean()), 1),
                           "p95": float(txt.str.len().quantile(0.95))},
    }
    if label_col and label_col in df.columns:
        dist = df[label_col].value_counts().to_dict()
        vals = list(dist.values())
        out["class_distribution"] = {str(k): int(v) for k, v in dist.items()}
        out["imbalance_ratio"] = round(min(vals) / max(vals), 3) if max(vals) else 0.0
    return out


def _ks_shift(a: pd.Series, b: pd.Series) -> float:
    from scipy.stats import ks_2samp

    return round(float(ks_2samp(a, b).statistic), 4)


def _label_noise_probe(path: Path, cap: int) -> dict[str, Any]:
    """Train a quick model; flag high-confidence disagreements as candidate label noise."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn.pipeline import Pipeline

    df = pd.read_csv(path)
    df["text"] = df["text"].map(clean_text)
    df["label"] = df["label"].astype(int)
    df = df[df["text"] != ""].reset_index(drop=True)
    if len(df) > cap:
        df = df.sample(n=cap, random_state=SEED).reset_index(drop=True)
    pipe = Pipeline([("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2)),
                     ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    proba = cross_val_predict(pipe, df["text"], df["label"], cv=3, method="predict_proba")[:, 1]
    pred = (proba >= 0.5).astype(int)
    confident_wrong = ((pred != df["label"]) & ((proba > 0.9) | (proba < 0.1))).mean()
    return {"rows": int(len(df)),
            "confident_misclassification_rate": round(float(confident_wrong), 4),
            "interpretation": "High-confidence disagreements are candidate mislabels or genuinely ambiguous items."}


def build(args: argparse.Namespace) -> dict[str, Any]:
    report: dict[str, Any] = {"datasets": {}, "distribution_shift": {}, "label_noise": {}}
    inherited = DATA_DIR.parent

    specs = [
        ("fake_reviews_v2", inherited / "fake_reviews.csv", "label"),
        ("fake_review_cross", DATA_DIR / "fake_review_cross.csv", "label"),
        ("amazon_polarity", DATA_DIR / "amazon_polarity.csv", "label"),
        ("yelp_polarity", DATA_DIR / "yelp_polarity.csv", "label"),
        ("imdb", DATA_DIR / "imdb.csv", "label"),
        ("sst2", DATA_DIR / "sst2.csv", "label"),
        ("amazon_reviews_v2", inherited / "amazon_reviews_sample.csv", None),
    ]
    for key, path, label_col in specs:
        if Path(path).exists():
            print(f"[quality] auditing {key}")
            report["datasets"][key] = _audit_text_dataset(Path(path), label_col, args.cap)

    # distribution shift: text length, amazon vs other sentiment domains
    def _lens(name: str) -> pd.Series | None:
        p = DATA_DIR / f"{name}.csv"
        if not p.exists():
            return None
        s = pd.read_csv(p, usecols=["text"]).sample(min(args.cap, 20000), random_state=SEED)["text"]
        return s.fillna("").astype(str).str.len()

    base = _lens("amazon_polarity")
    if base is not None:
        for tgt in ["yelp_polarity", "imdb", "sst2"]:
            t = _lens(tgt)
            if t is not None:
                report["distribution_shift"][f"amazon_vs_{tgt}_textlen_KS"] = _ks_shift(base, t)

    # label noise probe on the fake review training set
    fr = inherited / "fake_reviews.csv"
    if fr.exists():
        print("[quality] label-noise probe (fake reviews)")
        report["label_noise"]["fake_reviews_v2"] = _label_noise_probe(fr, min(args.cap, 20000))

    write_json(REPORTS_DIR / "data_quality_report.json", report)
    _write_md(report)
    _write_bias(report)
    print(f"[quality] -> {REPORTS_DIR / 'data_quality_report.md'}")
    return report


def _write_md(report: dict[str, Any]) -> None:
    L = ["# v3 Data Quality Report", "",
         "| Dataset | Rows | Empty | Exact dup | Near dup | Imbalance | Mean chars |",
         "|---|---|---|---|---|---|---|"]
    for k, d in report["datasets"].items():
        L.append(f"| {k} | {d['rows_sampled']} | {d['empty_text_rate']} | {d['exact_duplicate_rate']} | "
                 f"{d['near_duplicate_rate']} | {d.get('imbalance_ratio','-')} | {d['text_len_chars']['mean']} |")
    L += ["", "## Distribution shift (text length, KS vs Amazon)", ""]
    for k, v in report["distribution_shift"].items():
        L.append(f"- {k}: **{v}** (0 = identical, 1 = maximal shift)")
    if report["label_noise"]:
        L += ["", "## Label-noise probe", ""]
        for k, v in report["label_noise"].items():
            L.append(f"- {k}: confident-misclassification rate **{v['confident_misclassification_rate']}** — {v['interpretation']}")
    (REPORTS_DIR / "data_quality_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def _write_bias(report: dict[str, Any]) -> None:
    L = ["# v3 Bias Report", "",
         "## Class balance", ""]
    for k, d in report["datasets"].items():
        if "imbalance_ratio" in d:
            L.append(f"- **{k}**: imbalance ratio {d['imbalance_ratio']} (1.0 = perfectly balanced).")
    L += ["", "## Domain & coverage bias", "",
          "- All text datasets are **English only** — non-English reviews are out of scope.",
          "- Sentiment weak labels in v2 (`amazon_reviews_v2`) are ~80% positive (star-rating skew); "
          "the v3 sentiment study uses balanced polarity corpora instead.",
          "- v2 risk metadata is dominated by a few categories (All Beauty / Toys), biasing category-median "
          "price baselines; v3 reports category-normalised features.",
          "- Fake-review corpora are product/hotel reviews; generalisation to other review types is untested.", "",
          "## Distribution shift", "",
          "Non-trivial text-length shift between Amazon and Yelp/IMDB/SST-2 (see data_quality_report.md) "
          "means a model trained on one domain will degrade on another — quantified in evaluation_report.md."]
    (REPORTS_DIR / "bias_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cap", type=int, default=40000)
    return p


if __name__ == "__main__":
    build(build_parser().parse_args())
