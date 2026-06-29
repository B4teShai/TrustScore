"""Dataset & feature analysis for Presentation 2.

Produces per-dataset statistics (sizes, class balance, missing values, descriptive stats,
numeric correlations) and feature-importance evidence, saving JSON tables under
ml/reports/v2/ and PNG figures under ml/reports/v2/figures/. The metadata feature-importance
plot is the headline: it shows seller/price labels are driven by numeric fields, motivating
the v2 numeric+text models.
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
from sklearn.ensemble import RandomForestClassifier  # noqa: E402

try:
    from ml.training.common import clean_text, parse_float
    from ml.training.train_risk_models import price_label, seller_label
    from ml.training.v2.train_sentiment_model_v2 import sentiment_label_from_rating
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from common import clean_text, parse_float  # type: ignore
    from train_risk_models import price_label, seller_label  # type: ignore
    from v2.train_sentiment_model_v2 import sentiment_label_from_rating  # type: ignore


TEAL = "#1aa6a6"
CORAL = "#ff6f61"
PALETTE = [TEAL, CORAL, "#3d5a80", "#ee9b00", "#7b2cbf"]


def _data() -> Path:
    return Path(__file__).resolve().parents[2] / "data"


def _bar(counts: dict[str, Any], title: str, path: Path, xlabel: str = "") -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    keys = [str(k) for k in counts]
    ax.bar(keys, list(counts.values()), color=PALETTE[: len(keys)])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def analyse_fake(report: dict[str, Any], fig_dir: Path) -> None:
    frame = pd.read_csv(_data() / "fake_reviews.csv")
    frame["text"] = frame["text"].map(clean_text)
    frame["len"] = frame["text"].str.split().map(len)
    counts = frame["label"].astype(str).value_counts().to_dict()
    report["fake_reviews"] = {
        "rows": int(len(frame)),
        "features": list(frame.columns),
        "missing_values": frame.isna().sum().to_dict(),
        "class_distribution": {str(k): int(v) for k, v in counts.items()},
        "label_meaning": {"0": "original/authentic", "1": "computer-generated/fake"},
        "text_word_length": {
            "mean": round(float(frame["len"].mean()), 1),
            "median": float(frame["len"].median()),
            "p95": float(frame["len"].quantile(0.95)),
        },
    }
    _bar({str(k): int(v) for k, v in counts.items()}, "Fake-review class balance", fig_dir / "fake_class_balance.png", "label")


def analyse_reviews(report: dict[str, Any], fig_dir: Path) -> None:
    frame = pd.read_csv(_data() / "amazon_reviews_sample.csv")
    frame["label"] = frame["rating"].map(sentiment_label_from_rating)
    rating_counts = frame["rating"].round().value_counts().sort_index().to_dict()
    senti_counts = frame["label"].value_counts().to_dict()
    report["amazon_reviews"] = {
        "rows": int(len(frame)),
        "features": list(frame.columns),
        "missing_values": frame.isna().sum().to_dict(),
        "rating_distribution": {str(int(k)): int(v) for k, v in rating_counts.items()},
        "sentiment_distribution": {str(k): int(v) for k, v in senti_counts.items() if pd.notna(k)},
        "label_rule": "rating<=2 -> negative; 3 -> neutral; >=4 -> positive",
    }
    _bar(report["amazon_reviews"]["sentiment_distribution"], "Weak sentiment label balance", fig_dir / "sentiment_class_balance.png", "class")


def analyse_meta(report: dict[str, Any], fig_dir: Path, seed: int) -> None:
    frame = pd.read_csv(_data() / "amazon_meta_sample.csv")
    frame["main_category"] = frame.get("main_category", frame.get("source_category", "unknown"))
    frame["price_float"] = frame["price"].map(parse_float)
    frame["avg_rating"] = frame["average_rating"].map(lambda v: parse_float(v, np.nan))
    frame["n_ratings"] = frame["rating_number"].map(lambda v: parse_float(v, np.nan))
    frame["seller_label"] = frame.apply(seller_label, axis=1)
    medians = frame[frame["price_float"] > 0].groupby("main_category")["price_float"].median().to_dict()
    frame["price_label"] = frame.apply(
        lambda r: price_label(r["price_float"], medians.get(r["main_category"], float(np.nan))), axis=1
    )

    numeric = frame[["avg_rating", "n_ratings", "price_float"]].replace([np.inf, -np.inf], np.nan)
    corr = numeric.corr().round(3).fillna(0).to_dict()
    report["amazon_meta"] = {
        "rows": int(len(frame)),
        "features": list(frame.columns),
        "missing_values": {k: int(v) for k, v in frame.isna().sum().to_dict().items()},
        "numeric_describe": json.loads(numeric.describe().round(3).to_json()),
        "numeric_correlation": corr,
        "seller_label_distribution": {str(k): int(v) for k, v in frame["seller_label"].value_counts().to_dict().items()},
        "price_label_distribution": {str(k): int(v) for k, v in frame["price_label"].value_counts().to_dict().items()},
        "category_counts": {str(k): int(v) for k, v in frame["main_category"].value_counts().to_dict().items()},
    }
    _bar(report["amazon_meta"]["seller_label_distribution"], "Seller-reliability label balance", fig_dir / "seller_class_balance.png", "class")
    _bar(report["amazon_meta"]["price_label_distribution"], "Price-safety label balance", fig_dir / "price_class_balance.png", "class")

    # Headline figure: numeric features dominate the seller label.
    feats = frame[["avg_rating", "n_ratings", "price_float"]].fillna(0.0)
    rf = RandomForestClassifier(n_estimators=150, min_samples_leaf=2, class_weight="balanced", random_state=seed, n_jobs=-1)
    rf.fit(feats, frame["seller_label"])
    importances = dict(zip(["avg_rating", "n_ratings", "price"], [round(float(x), 4) for x in rf.feature_importances_]))
    report["amazon_meta"]["seller_label_numeric_feature_importance"] = importances
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(list(importances.keys()), list(importances.values()), color=TEAL)
    ax.set_title("Numeric feature importance for seller label (RF)")
    ax.set_xlabel("importance")
    fig.tight_layout()
    fig.savefig(fig_dir / "seller_numeric_feature_importance.png", dpi=130)
    plt.close(fig)


def build(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {}
    analyse_fake(report, fig_dir)
    analyse_reviews(report, fig_dir)
    analyse_meta(report, fig_dir, args.seed)
    (out_dir / "dataset_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[dataset-report] wrote {out_dir / 'dataset_report.json'} and figures in {fig_dir}")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parents[2] / "reports" / "v2"))
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    build(build_parser().parse_args())
