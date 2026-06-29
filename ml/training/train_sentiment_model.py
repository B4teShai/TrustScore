"""Train a fast review sentiment model from Amazon-style star labels."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
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


DATASET_NAME = "McAuley-Lab/Amazon-Reviews-2023"
DATASET_URL = "https://amazon-reviews-2023.github.io/main.html"
DATASET_HF_ROOT = f"hf://datasets/{DATASET_NAME}"


def amazon_review_file(category: str) -> str:
    return f"{DATASET_HF_ROOT}/raw/review_categories/{category}.jsonl"


def sentiment_label_from_rating(value: object) -> str | None:
    rating = parse_float(value, default=-1)
    if rating < 1:
        return None
    if rating <= 2:
        return "negative"
    if rating < 4:
        return "neutral"
    return "positive"


def load_sentiment_dataset(args: argparse.Namespace) -> tuple[pd.DataFrame, DatasetInfo]:
    if args.local_csv:
        frame = load_local_csv(args.local_csv)
        source = str(Path(args.local_csv))
        name = Path(args.local_csv).name
    else:
        frames = []
        for category in args.categories:
            frame = load_hf_data_files(
                "json",
                [amazon_review_file(category)],
                split=args.split,
                sample_size=args.sample_size,
                seed=args.seed,
            )
            frames.append(frame)
        frame = pd.concat(frames, ignore_index=True)
        source = "huggingface"
        name = f"{DATASET_NAME}:{','.join(args.categories)}"

    require_columns(frame, ("rating", "text"))
    frame = frame.copy()
    title = (
        frame["title"].fillna("").astype(str)
        if "title" in frame.columns
        else pd.Series([""] * len(frame))
    )
    frame["text"] = (title.fillna("").astype(str) + " " + frame["text"].fillna("").astype(str)).map(
        clean_text
    )
    frame["label"] = frame["rating"].map(sentiment_label_from_rating)
    frame = frame[(frame["text"] != "") & frame["label"].notna()].reset_index(drop=True)
    frame = sample_frame(frame, args.sample_size, args.seed)
    return frame, DatasetInfo(name=name, source=source, url=DATASET_URL, row_count=len(frame))


def train_sentiment_model(args: argparse.Namespace) -> dict[str, Any]:
    frame, dataset_info = load_sentiment_dataset(args)
    train_frame, test_frame = safe_train_test_split(frame, "label", args.test_size, args.seed)

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
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=args.max_iter,
                    random_state=args.seed,
                ),
            ),
        ]
    )
    pipeline.fit(train_frame["text"], train_frame["label"])
    predictions = pipeline.predict(test_frame["text"])
    metrics = multiclass_metrics(test_frame["label"], predictions)

    output_dir = Path(args.output_dir)
    artifacts = {
        "pipeline": str(output_dir / "sentiment_tfidf_logreg.joblib"),
        "metadata": str(output_dir / "sentiment_model_metadata.json"),
    }
    params = {
        "test_size": args.test_size,
        "seed": args.seed,
        "sample_size": args.sample_size,
        "categories": args.categories,
        "max_features": args.max_features,
        "max_ngram": args.max_ngram,
        "min_df": args.min_df,
        "max_iter": args.max_iter,
    }
    metadata = metadata_payload(
        model_name="sentiment_tfidf_logistic_regression",
        model_version=args.model_version,
        dataset=dataset_info,
        metrics=metrics,
        params=params,
        artifacts=artifacts,
        limitations=[
            "Sentiment labels are weak labels derived from star ratings.",
            "This artifact is for fast local training; transformer fine-tuning can replace it later.",
        ],
    )

    if not args.no_save:
        dump_joblib(Path(artifacts["pipeline"]), pipeline)
        write_json(Path(artifacts["metadata"]), metadata)

    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-csv")
    parser.add_argument("--categories", nargs="+", default=["All_Beauty"])
    parser.add_argument("--split", default="full")
    parser.add_argument("--output-dir", default=str(default_artifacts_dir()))
    parser.add_argument("--model-version", default="0.1.0")
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=40000)
    parser.add_argument("--max-ngram", type=int, default=2)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--no-save", action="store_true")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    result = train_sentiment_model(args)
    print(result["metrics"])
