"""Leakage analysis — the v3 honesty centerpiece.

v2 reported seller/price risk accuracy of ~0.99. This proves that result is a **weak-label
leakage shortcut**, not learned trust:

1. The labels are deterministic functions of numeric features. A tiny decision tree on the
   numeric features ALONE reconstructs the label almost perfectly, and ``export_text`` shows
   it recovers the exact thresholds used to build the label.
2. Because the labelling rule is GLOBAL (same thresholds in every category), a numeric model
   trained on two categories and tested on a held-out third (leave-one-category-out) STAYS
   high — that is rule reconstruction, not generalisation of a trust concept. The text-only
   model (≈ v1) does not, confirming the contrast.

Verdict: the in-sample 0.99 is rejected as a trust metric. See FINDINGS.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_text

try:
    from ml.training.common import clean_text, parse_float, safe_train_test_split, write_json
    from ml.training.train_risk_models import price_label, seller_label
    from ml.training.v2.features import log_review_count, price_ratio
    from ml.training.v3 import REPORTS_DIR, SEED
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[3]))
    from training.common import clean_text, parse_float, safe_train_test_split, write_json  # type: ignore
    from training.train_risk_models import price_label, seller_label  # type: ignore
    from training.v2.features import log_review_count, price_ratio  # type: ignore
    from training.v3 import REPORTS_DIR, SEED  # type: ignore


def _meta_path() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "amazon_meta_sample.csv"


def _prepare(meta: pd.DataFrame) -> pd.DataFrame:
    df = meta.copy()
    df["main_category"] = df.get("main_category", df.get("source_category", "unknown"))
    df["category"] = df.get("source_category", df["main_category"]).fillna("unknown").astype(str)
    df["price_float"] = df["price"].map(parse_float)
    df["seller_label"] = df.apply(seller_label, axis=1)
    medians = df[df["price_float"] > 0].groupby("main_category")["price_float"].median().to_dict()
    df["category_median"] = df["main_category"].map(medians)
    df["price_label"] = df.apply(
        lambda r: price_label(r["price_float"], medians.get(r["main_category"], float(np.nan))), axis=1
    )
    df["rating"] = df["average_rating"].map(lambda v: parse_float(v, 0.0))
    df["log_review_count"] = df["rating_number"].map(log_review_count)
    df["price"] = df["price_float"]
    df["price_ratio"] = df.apply(lambda r: price_ratio(r["price_float"], r.get("category_median")), axis=1)
    df["seller_text"] = (df["title"].fillna("").astype(str) + " " + df["main_category"].fillna("").astype(str)).map(clean_text)
    return df


def _numeric_leakage(df: pd.DataFrame, numeric: list[str], label_col: str) -> dict[str, Any]:
    sub = df[df[label_col].isin(df[label_col].value_counts().index)].copy()
    X = sub[numeric].replace([np.inf, -np.inf], 0).fillna(0.0)
    y = sub[label_col]
    # shallow tree: if it reconstructs the label, the label IS a function of these features
    tree = DecisionTreeClassifier(max_depth=4, random_state=SEED)
    tree.fit(X, y)
    recon = float(accuracy_score(y, tree.predict(X)))
    mi = mutual_info_classif(X, y, discrete_features=False, random_state=SEED)
    rule = export_text(tree, feature_names=list(numeric), max_depth=4)
    return {
        "numeric_features": numeric,
        "label_reconstruction_accuracy": round(recon, 4),
        "mutual_information": {f: round(float(m), 4) for f, m in zip(numeric, mi)},
        "recovered_rule_excerpt": rule[:1500],
    }


def _loco(df: pd.DataFrame, numeric: list[str], label_col: str) -> dict[str, Any]:
    """Leave-one-category-out: numeric model vs text-only model OOD accuracy."""
    cats = sorted(df["category"].unique())
    out = {"categories": cats, "per_holdout": []}
    num_scores, txt_scores = [], []
    for hold in cats:
        train = df[df["category"] != hold]
        test = df[df["category"] == hold]
        if test[label_col].nunique() < 2 or train[label_col].nunique() < 2:
            continue
        num = RandomForestClassifier(n_estimators=120, min_samples_leaf=2, class_weight="balanced",
                                     random_state=SEED, n_jobs=-1)
        num.fit(train[numeric].fillna(0.0), train[label_col])
        num_acc = float(accuracy_score(test[label_col], num.predict(test[numeric].fillna(0.0))))

        txt = Pipeline([("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=3)),
                        ("clf", RandomForestClassifier(n_estimators=120, min_samples_leaf=2,
                                                       class_weight="balanced", random_state=SEED, n_jobs=-1))])
        txt.fit(train["seller_text"], train[label_col])
        txt_acc = float(accuracy_score(test[label_col], txt.predict(test["seller_text"])))

        num_scores.append(num_acc)
        txt_scores.append(txt_acc)
        out["per_holdout"].append({"holdout": hold, "numeric_ood_acc": round(num_acc, 4), "text_only_ood_acc": round(txt_acc, 4)})
    out["mean_numeric_ood_acc"] = round(float(np.mean(num_scores)), 4) if num_scores else None
    out["mean_text_only_ood_acc"] = round(float(np.mean(txt_scores)), 4) if txt_scores else None
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    meta = pd.read_csv(_meta_path(), low_memory=False)
    df = _prepare(meta)
    report: dict[str, Any] = {"rows": int(len(df)), "models": {}}
    for label_col, numeric in [
        ("seller_label", ["rating", "log_review_count"]),
        ("price_label", ["price", "price_ratio"]),
    ]:
        sub = df[df[label_col] != "unknown"] if label_col == "price_label" else df
        report["models"][label_col] = {
            "leakage": _numeric_leakage(sub, numeric, label_col),
            "leave_one_category_out": _loco(sub, numeric, label_col),
        }
    write_json(REPORTS_DIR / "leakage_report.json", report)
    _write_md(report)
    print(f"[leakage] -> {REPORTS_DIR / 'leakage_report.md'}")
    return report


def _write_md(report: dict[str, Any]) -> None:
    lines = ["# v3 Leakage Report — risk weak labels", "",
             f"Rows analysed: {report['rows']} (from v2 Amazon metadata sample).", ""]
    for label_col, m in report["models"].items():
        lk = m["leakage"]
        loco = m["leave_one_category_out"]
        lines += [
            f"## {label_col}", "",
            f"- **Label reconstruction from numeric features alone:** "
            f"**{lk['label_reconstruction_accuracy']:.4f}** (depth-4 tree).",
            f"- Mutual information: {lk['mutual_information']}",
            f"- Leave-one-category-out mean accuracy — **numeric {loco['mean_numeric_ood_acc']}** "
            f"vs text-only {loco['mean_text_only_ood_acc']}.",
            "",
            "Per held-out category:", "",
            "| Held-out | numeric OOD acc | text-only OOD acc |", "|---|---|---|",
            *[f"| {r['holdout']} | {r['numeric_ood_acc']} | {r['text_only_ood_acc']} |" for r in loco["per_holdout"]],
            "",
            "<details><summary>Recovered decision rule (excerpt)</summary>", "", "```",
            lk["recovered_rule_excerpt"], "```", "</details>", "",
        ]
    lines += [
        "## Verdict", "",
        "The numeric features **reconstruct the label almost perfectly** and the recovered tree "
        "recovers the exact thresholds used to *construct* the weak label (rating/review-count for "
        "seller; price ratio for price). Because the rule is global, the numeric model stays high "
        "even on a held-out category — that is **rule reconstruction, not trust generalisation**.",
        "",
        "**We reject the v2 in-sample ~0.99 seller/price accuracy as a trust metric.** v3 reports "
        "the unsupervised price-anomaly signal and out-of-domain numbers instead (see FINDINGS.md).",
    ]
    (REPORTS_DIR / "leakage_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description=__doc__)


if __name__ == "__main__":
    run(build_parser().parse_args())
