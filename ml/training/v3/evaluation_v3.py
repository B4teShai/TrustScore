"""Phase 6 — real-world evaluation: out-of-domain, calibration, robustness.

The headline honesty metric is generalisation under distribution shift, not in-sample accuracy:
- Sentiment: train on Amazon polarity, test in-domain AND cross-domain (Yelp / IMDB / SST-2).
- Fake review: leave-one-category-out across product categories (genuine OOD, same corpus).
- Risk: echoes the leakage report's leave-one-category-out (numeric vs text-only).
Plus calibration (Brier / ECE / reliability) and robustness to text perturbations.

Uses fast TF-IDF + LogisticRegression as the common probe so the OOD *gap* is the signal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, roc_auc_score  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

try:
    from ml.training.common import clean_text, write_json
    from ml.training.v3 import DATA_DIR, FIGURES_DIR, REPORTS_DIR, SEED
    from ml.training.v3.registry import log_experiment
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from training.common import clean_text, write_json  # type: ignore
    from training.v3 import DATA_DIR, FIGURES_DIR, REPORTS_DIR, SEED  # type: ignore
    from training.v3.registry import log_experiment  # type: ignore


def _probe() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)),
    ])


def _ece(y_true: np.ndarray, prob: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (prob > lo) & (prob <= hi)
        if m.sum() == 0:
            continue
        ece += (m.mean()) * abs(y_true[m].mean() - prob[m].mean())
    return float(ece)


def _load(name: str, cap: int) -> pd.DataFrame | None:
    path = DATA_DIR / f"{name}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "label" not in df.columns:
        return None
    df = df[["text", "label"]].dropna()
    df["label"] = df["label"].astype(int)
    if len(df) > cap:
        df = df.sample(n=cap, random_state=SEED).reset_index(drop=True)
    return df


def sentiment_cross_domain(cap: int) -> dict[str, Any]:
    amazon = _load("amazon_polarity", cap)
    if amazon is None:
        return {"status": "amazon_polarity missing"}
    probe = _probe()
    tr = amazon.sample(frac=0.8, random_state=SEED)
    held = amazon.drop(tr.index)
    probe.fit(tr["text"], tr["label"])

    def score(df: pd.DataFrame) -> dict[str, Any]:
        prob = probe.predict_proba(df["text"])[:, 1]
        pred = (prob >= 0.5).astype(int)
        return {"n": int(len(df)), "accuracy": round(float(accuracy_score(df["label"], pred)), 4),
                "f1": round(float(f1_score(df["label"], pred)), 4),
                "roc_auc": round(float(roc_auc_score(df["label"], prob)), 4)}

    out = {"train": "amazon_polarity", "in_domain": score(held), "cross_domain": {}}
    for tgt in ["yelp_polarity", "imdb", "sst2"]:
        df = _load(tgt, cap)
        if df is not None:
            out["cross_domain"][tgt] = score(df)
    return out


def fake_leave_one_category_out() -> dict[str, Any]:
    path = DATA_DIR / "fake_review_cross.csv"
    if not path.exists():
        return {"status": "fake_review_cross missing"}
    df = pd.read_csv(path)
    df["text"] = df["text"].map(clean_text)
    df = df[(df["text"] != "")].reset_index(drop=True)
    cats = df["category"].value_counts()
    cats = cats[cats >= 500].index.tolist()
    per = []
    for hold in cats:
        tr = df[df["category"] != hold]
        te = df[df["category"] == hold]
        if te["label"].nunique() < 2:
            continue
        probe = _probe()
        probe.fit(tr["text"], tr["label"])
        prob = probe.predict_proba(te["text"])[:, 1]
        pred = (prob >= 0.5).astype(int)
        per.append({"holdout": hold, "n": int(len(te)),
                    "accuracy": round(float(accuracy_score(te["label"], pred)), 4),
                    "roc_auc": round(float(roc_auc_score(te["label"], prob)), 4)})
    # in-domain reference (random split)
    probe = _probe()
    tr = df.sample(frac=0.8, random_state=SEED)
    te = df.drop(tr.index)
    probe.fit(tr["text"], tr["label"])
    prob = probe.predict_proba(te["text"])[:, 1]
    indom = {"accuracy": round(float(accuracy_score(te["label"], (prob >= 0.5).astype(int))), 4),
             "roc_auc": round(float(roc_auc_score(te["label"], prob)), 4)}
    mean_ood = round(float(np.mean([p["accuracy"] for p in per])), 4) if per else None
    return {"in_domain_random_split": indom, "mean_ood_accuracy": mean_ood, "per_category": per}


def calibration_and_robustness(cap: int) -> dict[str, Any]:
    path = Path(__file__).resolve().parents[2] / "data" / "fake_reviews.csv"
    df = pd.read_csv(path)
    df["text"] = df["text"].map(clean_text)
    df["label"] = df["label"].astype(int)
    df = df[df["text"] != ""].reset_index(drop=True)
    if len(df) > cap:
        df = df.sample(n=cap, random_state=SEED).reset_index(drop=True)
    tr = df.sample(frac=0.8, random_state=SEED)
    te = df.drop(tr.index)

    raw = _probe().fit(tr["text"], tr["label"])
    raw_prob = raw.predict_proba(te["text"])[:, 1]
    cal = CalibratedClassifierCV(_probe(), cv=3, method="isotonic").fit(tr["text"], tr["label"])
    cal_prob = cal.predict_proba(te["text"])[:, 1]
    y = te["label"].to_numpy()

    calibration = {
        "raw": {"brier": round(float(brier_score_loss(y, raw_prob)), 4), "ece": round(_ece(y, raw_prob), 4)},
        "isotonic": {"brier": round(float(brier_score_loss(y, cal_prob)), 4), "ece": round(_ece(y, cal_prob), 4)},
    }
    _reliability_fig(y, raw_prob, cal_prob)

    # robustness: perturb test text, measure accuracy delta
    base_acc = float(accuracy_score(y, (raw_prob >= 0.5).astype(int)))
    perts = {
        "lowercase": te["text"].str.lower(),
        "truncate_50_chars": te["text"].str.slice(0, 50),
        "drop_vowels_10pct": te["text"].map(_drop_some_vowels),
    }
    robustness = {"baseline_accuracy": round(base_acc, 4), "perturbed": {}}
    for pname, ptext in perts.items():
        acc = float(accuracy_score(y, (raw.predict_proba(ptext)[:, 1] >= 0.5).astype(int)))
        robustness["perturbed"][pname] = {"accuracy": round(acc, 4), "delta": round(acc - base_acc, 4)}
    return {"calibration": calibration, "robustness": robustness}


def _drop_some_vowels(text: str) -> str:
    out, n = [], 0
    for ch in text:
        n += 1
        if ch.lower() in "aeiou" and n % 10 == 0:
            continue
        out.append(ch)
    return "".join(out)


def _reliability_fig(y, raw_prob, cal_prob) -> None:
    from sklearn.calibration import calibration_curve

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
    for prob, name, c in [(raw_prob, "raw", "#ff6f61"), (cal_prob, "isotonic", "#1aa6a6")]:
        ft, mp = calibration_curve(y, prob, n_bins=10, strategy="quantile")
        ax.plot(mp, ft, "o-", color=c, label=name)
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed frequency")
    ax.set_title("Fake-review model calibration")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "calibration_reliability.png", dpi=130)
    plt.close(fig)


def _cross_domain_fig(sentiment: dict[str, Any]) -> None:
    if "in_domain" not in sentiment:
        return
    labels = ["amazon (in-dom)"] + list(sentiment["cross_domain"].keys())
    accs = [sentiment["in_domain"]["accuracy"]] + [v["accuracy"] for v in sentiment["cross_domain"].values()]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#1aa6a6"] + ["#ff6f61"] * (len(labels) - 1)
    ax.bar(labels, accs, color=colors)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("accuracy")
    ax.set_title("Sentiment: in-domain vs cross-domain (train=Amazon)")
    for i, a in enumerate(accs):
        ax.text(i, a + 0.01, f"{a:.2f}", ha="center", fontsize=9)
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "sentiment_cross_domain.png", dpi=130)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, Any]:
    report: dict[str, Any] = {}
    print("[eval] sentiment cross-domain...")
    report["sentiment_cross_domain"] = sentiment_cross_domain(args.cap)
    _cross_domain_fig(report["sentiment_cross_domain"])
    print("[eval] fake review leave-one-category-out...")
    report["fake_review_ood"] = fake_leave_one_category_out()
    print("[eval] calibration + robustness...")
    report["fake_review_calibration_robustness"] = calibration_and_robustness(args.cap)
    # echo risk leakage LOCO if present
    lk = REPORTS_DIR / "leakage_report.json"
    if lk.exists():
        report["risk_leakage_reference"] = json.loads(lk.read_text())["models"]

    write_json(REPORTS_DIR / "evaluation_report.json", report)
    _write_md(report)
    log_experiment(phase="6_evaluation", name="ood_calibration_robustness", params={"cap": args.cap},
                   metrics={"sentiment_cross_domain": report["sentiment_cross_domain"].get("cross_domain", {})})
    print(f"[eval] -> {REPORTS_DIR / 'evaluation_report.md'}")
    return report


def _write_md(report: dict[str, Any]) -> None:
    L = ["# v3 Real-World Evaluation", "",
         "Headline metric: **generalisation under distribution shift**, not in-sample accuracy.", ""]
    sd = report.get("sentiment_cross_domain", {})
    if "in_domain" in sd:
        L += ["## Sentiment — cross-domain (train = Amazon polarity)", "",
              "| Target | n | Acc | F1 | ROC-AUC |", "|---|---|---|---|---|",
              f"| amazon (in-domain) | {sd['in_domain']['n']} | {sd['in_domain']['accuracy']} | {sd['in_domain']['f1']} | {sd['in_domain']['roc_auc']} |"]
        for t, v in sd["cross_domain"].items():
            L.append(f"| {t} (OOD) | {v['n']} | {v['accuracy']} | {v['f1']} | {v['roc_auc']} |")
        L += ["", "The accuracy drop from Amazon to Yelp/IMDB/SST-2 quantifies real domain shift.", ""]
    fo = report.get("fake_review_ood", {})
    if "mean_ood_accuracy" in fo:
        L += ["## Fake review — leave-one-category-out (genuine OOD)", "",
              f"- In-domain random split: acc **{fo['in_domain_random_split']['accuracy']}**, "
              f"AUC {fo['in_domain_random_split']['roc_auc']}.",
              f"- Mean leave-one-category-out accuracy: **{fo['mean_ood_accuracy']}**.", "",
              "| Held-out category | n | Acc | ROC-AUC |", "|---|---|---|---|",
              *[f"| {p['holdout']} | {p['n']} | {p['accuracy']} | {p['roc_auc']} |" for p in fo["per_category"]], ""]
    cr = report.get("fake_review_calibration_robustness", {})
    if cr:
        cal = cr["calibration"]; rob = cr["robustness"]
        L += ["## Calibration (fake-review probe)", "",
              f"- Raw: Brier {cal['raw']['brier']}, ECE {cal['raw']['ece']}.",
              f"- Isotonic: Brier {cal['isotonic']['brier']}, ECE {cal['isotonic']['ece']}.",
              "- Reliability diagram: `figures/calibration_reliability.png`.", "",
              "## Robustness (text perturbations)", "",
              f"Baseline accuracy {rob['baseline_accuracy']}.", "",
              "| Perturbation | Acc | Δ |", "|---|---|---|",
              *[f"| {k} | {v['accuracy']} | {v['delta']} |" for k, v in rob["perturbed"].items()], ""]
    if "risk_leakage_reference" in report:
        L += ["## Risk — out-of-domain (from leakage report)", "",
              "Numeric risk models keep ~1.0 accuracy leave-one-category-out because the label is a "
              "global rule of the numeric features (leakage). Text-only collapses. See `leakage_report.md`. "
              "v3 therefore treats risk via the unsupervised price-anomaly signal, not the weak-label classifier.", ""]
    (REPORTS_DIR / "evaluation_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cap", type=int, default=40000)
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
