"""Train weak-label risk models for seller, price, and policy components."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

try:
    from ml.training.common import (
        DatasetInfo,
        clean_text,
        default_artifacts_dir,
        dump_joblib,
        load_local_csv,
        load_hf_data_files,
        metadata_payload,
        multiclass_metrics,
        parse_float,
        require_columns,
        safe_train_test_split,
        sample_frame,
        write_json,
    )
except ModuleNotFoundError:
    from common import (  # type: ignore[no-redef]
        DatasetInfo,
        clean_text,
        default_artifacts_dir,
        dump_joblib,
        load_local_csv,
        load_hf_data_files,
        metadata_payload,
        multiclass_metrics,
        parse_float,
        require_columns,
        safe_train_test_split,
        sample_frame,
        write_json,
    )


AMAZON_DATASET_NAME = "McAuley-Lab/Amazon-Reviews-2023"
AMAZON_DATASET_URL = "https://amazon-reviews-2023.github.io/main.html"
AMAZON_DATASET_HF_ROOT = f"hf://datasets/{AMAZON_DATASET_NAME}"
TARGET_DATASET_URL = "https://github.com/luminati-io/Target-dataset-samples"


def amazon_meta_files(category: str) -> str:
    return f"{AMAZON_DATASET_HF_ROOT}/raw_meta_{category}/full-*.parquet"


def seller_label(row: pd.Series) -> str:
    rating = parse_float(row.get("average_rating"), default=0)
    count = parse_float(row.get("rating_number"), default=0)
    if rating >= 4.2 and count >= 50:
        return "reliable"
    if rating < 3.5 or count < 5:
        return "weak"
    return "mixed"


def price_label(price: float, category_median: float) -> str:
    if price <= 0 or category_median <= 0:
        return "unknown"
    ratio = price / category_median
    if ratio < 0.5:
        return "suspicious_low"
    if ratio > 1.8:
        return "high_price"
    return "normal"


def policy_label(policy: object) -> str:
    text = clean_text(policy)
    if not text:
        return "unclear"
    has_return = "return" in text or "refund" in text
    has_period = bool(__import__("re").search(r"\b\d+\s*-?\s*(day|days|week|weeks|month|months)\b", text))
    has_warranty = "warranty" in text or "exchange" in text or "replacement" in text
    if has_return and has_period and has_warranty:
        return "clear"
    if has_return or has_period:
        return "partial"
    return "unclear"


def load_amazon_meta(args: argparse.Namespace) -> tuple[pd.DataFrame, DatasetInfo]:
    if args.amazon_meta_csv:
        frame = load_local_csv(args.amazon_meta_csv)
        source = str(Path(args.amazon_meta_csv))
        name = Path(args.amazon_meta_csv).name
    else:
        frames = []
        for category in args.categories:
            frame = load_hf_data_files(
                "parquet",
                [amazon_meta_files(category)],
                split=args.split,
                sample_size=args.sample_size,
                seed=args.seed,
            )
            frame["source_category"] = category
            frames.append(frame)
        frame = pd.concat(frames, ignore_index=True)
        source = "huggingface"
        name = f"{AMAZON_DATASET_NAME}:meta:{','.join(args.categories)}"

    require_columns(frame, ("title", "average_rating", "rating_number", "price"))
    frame = frame.copy()
    frame["main_category"] = frame.get("main_category", frame.get("source_category", "unknown"))
    frame["price_float"] = frame["price"].map(parse_float)
    frame = sample_frame(frame, args.sample_size, args.seed)
    return frame, DatasetInfo(name=name, source=source, url=AMAZON_DATASET_URL, row_count=len(frame))


def load_policy_frame(args: argparse.Namespace) -> tuple[pd.DataFrame, DatasetInfo] | tuple[None, None]:
    if not args.policy_csv:
        return None, None
    frame = load_local_csv(args.policy_csv)
    column = args.policy_column
    require_columns(frame, (column,))
    frame = frame.copy()
    frame["policy_text"] = frame[column].map(clean_text)
    frame["policy_label"] = frame[column].map(policy_label)
    frame = frame[frame["policy_text"] != ""].reset_index(drop=True)
    frame = sample_frame(frame, args.sample_size, args.seed)
    return frame, DatasetInfo(
        name=Path(args.policy_csv).name,
        source=str(Path(args.policy_csv)),
        url=TARGET_DATASET_URL,
        row_count=len(frame),
    )


def train_risk_models(args: argparse.Namespace) -> dict[str, Any]:
    meta_frame, meta_info = load_amazon_meta(args)
    meta_frame = meta_frame.copy()
    meta_frame["seller_label"] = meta_frame.apply(seller_label, axis=1)
    category_medians = (
        meta_frame[meta_frame["price_float"] > 0]
        .groupby("main_category")["price_float"]
        .median()
        .to_dict()
    )
    meta_frame["price_label"] = meta_frame.apply(
        lambda row: price_label(
            row["price_float"],
            category_medians.get(row["main_category"], float(np.nan)),
        ),
        axis=1,
    )
    store_series = (
        meta_frame["store"].fillna("").astype(str)
        if "store" in meta_frame.columns
        else pd.Series([""] * len(meta_frame))
    )
    meta_frame["seller_text"] = (
        store_series
        + " "
        + meta_frame["title"].fillna("").astype(str)
        + " "
        + meta_frame["main_category"].fillna("").astype(str)
    ).map(clean_text)
    meta_frame["price_text"] = (
        meta_frame["title"].fillna("").astype(str)
        + " "
        + meta_frame["main_category"].fillna("").astype(str)
    ).map(clean_text)

    seller_model, seller_metrics = _fit_text_classifier(
        meta_frame[meta_frame["seller_text"] != ""],
        text_column="seller_text",
        label_column="seller_label",
        args=args,
    )
    price_model, price_metrics = _fit_text_classifier(
        meta_frame[(meta_frame["price_text"] != "") & (meta_frame["price_label"] != "unknown")],
        text_column="price_text",
        label_column="price_label",
        args=args,
    )

    policy_model = None
    policy_metrics: dict[str, Any] | None = None
    policy_info: DatasetInfo | None = None
    policy_frame, maybe_policy_info = load_policy_frame(args)
    if policy_frame is not None and maybe_policy_info is not None and len(policy_frame) >= 4:
        policy_info = maybe_policy_info
        policy_model, policy_metrics = _fit_text_classifier(
            policy_frame,
            text_column="policy_text",
            label_column="policy_label",
            args=args,
        )

    output_dir = Path(args.output_dir)
    artifacts = {
        "seller_model": str(output_dir / "seller_reliability_tfidf_rf.joblib"),
        "price_model": str(output_dir / "price_safety_tfidf_rf.joblib"),
        "policy_model": str(output_dir / "policy_clarity_tfidf_rf.joblib"),
        "metadata": str(output_dir / "risk_model_metadata.json"),
    }
    metrics = {
        "seller_reliability": seller_metrics,
        "price_safety": price_metrics,
        "policy_clarity": policy_metrics or {"status": "skipped_missing_policy_csv"},
    }
    metadata = metadata_payload(
        model_name="trustscore_weak_label_risk_models",
        model_version=args.model_version,
        dataset=meta_info,
        metrics=metrics,
        params={
            "test_size": args.test_size,
            "seed": args.seed,
            "sample_size": args.sample_size,
            "categories": args.categories,
            "policy_dataset": policy_info.__dict__ if policy_info else None,
        },
        artifacts=artifacts,
        limitations=[
            "Seller, price, and policy labels are weak labels derived from public metadata.",
            "Current backend still uses deterministic rule-based scoring until these artifacts are explicitly integrated.",
        ],
    )

    if not args.no_save:
        dump_joblib(Path(artifacts["seller_model"]), seller_model)
        dump_joblib(Path(artifacts["price_model"]), price_model)
        if policy_model is not None:
            dump_joblib(Path(artifacts["policy_model"]), policy_model)
        write_json(Path(artifacts["metadata"]), metadata)

    return metadata


def _fit_text_classifier(
    frame: pd.DataFrame,
    *,
    text_column: str,
    label_column: str,
    args: argparse.Namespace,
) -> tuple[Pipeline, dict[str, Any]]:
    train_frame, test_frame = safe_train_test_split(frame, label_column, args.test_size, args.seed)
    pipeline = Pipeline(
        [
            (
                "vectorizer",
                TfidfVectorizer(
                    max_features=args.max_features,
                    ngram_range=(1, args.max_ngram),
                    min_df=args.min_df,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=args.n_estimators,
                    min_samples_leaf=args.min_samples_leaf,
                    class_weight="balanced",
                    random_state=args.seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipeline.fit(train_frame[text_column], train_frame[label_column])
    predictions = pipeline.predict(test_frame[text_column])
    return pipeline, multiclass_metrics(test_frame[label_column], predictions)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amazon-meta-csv")
    parser.add_argument("--policy-csv")
    parser.add_argument("--policy-column", default="shipping_returns_policy")
    parser.add_argument("--categories", nargs="+", default=["All_Beauty"])
    parser.add_argument("--split", default="full")
    parser.add_argument("--output-dir", default=str(default_artifacts_dir()))
    parser.add_argument("--model-version", default="0.1.0")
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=20000)
    parser.add_argument("--max-ngram", type=int, default=2)
    parser.add_argument("--min-df", type=int, default=1)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--no-save", action="store_true")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    result = train_risk_models(args)
    print(result["metrics"])
