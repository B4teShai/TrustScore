"""Naive baselines for every TrustScore task, for the comparison table.

Provides the floor each model must beat: a majority-class DummyClassifier plus a simple
word-only TF-IDF + LogisticRegression baseline. Written to ml/reports/v2/baselines.json.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

try:
    from ml.training.common import (
        clean_text,
        multiclass_metrics,
        parse_float,
        safe_train_test_split,
        write_json,
    )
    from ml.training.train_risk_models import price_label, seller_label
    from ml.training.v2.train_sentiment_model_v2 import sentiment_label_from_rating
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from common import clean_text, multiclass_metrics, parse_float, safe_train_test_split, write_json  # type: ignore
    from train_risk_models import price_label, seller_label  # type: ignore
    from v2.train_sentiment_model_v2 import sentiment_label_from_rating  # type: ignore


def _default_data() -> Path:
    return Path(__file__).resolve().parents[2] / "data"


def _eval_task(frame: pd.DataFrame, text_col: str, label_col: str, seed: int, cap: int) -> dict[str, Any]:
    if len(frame) > cap:
        frame = frame.sample(n=cap, random_state=seed).reset_index(drop=True)
    train, test = safe_train_test_split(frame, label_col, 0.2, seed)

    dummy = DummyClassifier(strategy="most_frequent", random_state=seed)
    dummy.fit(np.zeros((len(train), 1)), train[label_col])
    dummy_metrics = multiclass_metrics(test[label_col], dummy.predict(np.zeros((len(test), 1))))

    logreg = Pipeline([
        ("vectorizer", TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000)),
    ])
    logreg.fit(train[text_col], train[label_col])
    logreg_metrics = multiclass_metrics(test[label_col], logreg.predict(test[text_col]))
    return {
        "n_rows": int(len(frame)),
        "majority_class_baseline": dummy_metrics,
        "tfidf_logreg_baseline": logreg_metrics,
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    data = _default_data()
    out: dict[str, Any] = {}

    fake = pd.read_csv(data / "fake_reviews.csv")
    fake["text"] = fake["text"].map(clean_text)
    fake["label"] = fake["label"].astype(str)
    fake = fake[fake["text"] != ""].reset_index(drop=True)
    out["fake_review"] = _eval_task(fake, "text", "label", args.seed, args.cap)

    rev = pd.read_csv(data / "amazon_reviews_sample.csv")
    title = rev["title"].fillna("").astype(str) if "title" in rev.columns else ""
    rev["text"] = (title + " " + rev["text"].fillna("").astype(str)).map(clean_text)
    rev["label"] = rev["rating"].map(sentiment_label_from_rating)
    rev = rev[(rev["text"] != "") & rev["label"].notna()].reset_index(drop=True)
    out["sentiment"] = _eval_task(rev, "text", "label", args.seed, args.cap)

    meta = pd.read_csv(data / "amazon_meta_sample.csv")
    meta["main_category"] = meta.get("main_category", meta.get("source_category", "unknown"))
    meta["price_float"] = meta["price"].map(parse_float)
    meta["seller_label"] = meta.apply(seller_label, axis=1)
    medians = meta[meta["price_float"] > 0].groupby("main_category")["price_float"].median().to_dict()
    meta["price_label"] = meta.apply(
        lambda r: price_label(r["price_float"], medians.get(r["main_category"], float(np.nan))), axis=1
    )
    meta["seller_text"] = (meta["title"].fillna("").astype(str) + " " + meta["main_category"].astype(str)).map(clean_text)
    seller_df = meta[meta["seller_text"] != ""]
    out["seller_reliability"] = _eval_task(seller_df, "seller_text", "seller_label", args.seed, args.cap)
    price_df = meta[(meta["seller_text"] != "") & (meta["price_label"] != "unknown")]
    out["price_safety"] = _eval_task(price_df, "seller_text", "price_label", args.seed, args.cap)

    report_dir = Path(args.output_dir)
    write_json(report_dir / "baselines.json", out)
    print(f"[baselines] wrote {report_dir / 'baselines.json'}")
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parents[2] / "reports" / "v2"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cap", type=int, default=40000, help="row cap per task for fast baselines")
    return parser


if __name__ == "__main__":
    build(build_parser().parse_args())
