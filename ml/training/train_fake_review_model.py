"""Train the fake-review classifier used by backend artifact inference."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from ml.training.common import (
        DatasetInfo,
        binary_metrics,
        clean_text,
        default_artifacts_dir,
        dump_joblib,
        load_local_csv,
        metadata_payload,
        require_columns,
        safe_train_test_split,
        sample_frame,
        write_json,
    )
except ModuleNotFoundError:
    from common import (  # type: ignore[no-redef]
        DatasetInfo,
        binary_metrics,
        clean_text,
        default_artifacts_dir,
        dump_joblib,
        load_local_csv,
        metadata_payload,
        require_columns,
        safe_train_test_split,
        sample_frame,
        write_json,
    )


DATASET_NAME = "theArijitDas/Fake-Reviews-Dataset"
DATASET_URL = "https://huggingface.co/datasets/theArijitDas/Fake-Reviews-Dataset"


def load_fake_review_dataset(args: argparse.Namespace) -> tuple[pd.DataFrame, DatasetInfo]:
    if args.local_csv:
        frame = load_local_csv(args.local_csv)
        source = str(Path(args.local_csv))
        url = None
        name = Path(args.local_csv).name
    else:
        from datasets import load_dataset

        dataset = load_dataset(args.dataset_name, split=args.split)
        frame = dataset.to_pandas()
        source = "huggingface"
        url = DATASET_URL
        name = args.dataset_name

    require_columns(frame, ("text", "label"))
    frame = frame.copy()
    frame["text"] = frame["text"].map(clean_text)
    frame["label"] = frame["label"].astype(int)
    frame = frame[(frame["text"] != "") & frame["label"].isin([0, 1])].reset_index(drop=True)
    frame = sample_frame(frame, args.sample_size, args.seed)
    return frame, DatasetInfo(name=name, source=source, url=url, row_count=len(frame))


def train_fake_review_model(args: argparse.Namespace) -> dict[str, Any]:
    frame, dataset_info = load_fake_review_dataset(args)
    train_frame, test_frame = safe_train_test_split(frame, "label", args.test_size, args.seed)

    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        ngram_range=(1, args.max_ngram),
        min_df=args.min_df,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    x_train = vectorizer.fit_transform(train_frame["text"])
    x_test = vectorizer.transform(test_frame["text"])

    classifier = RandomForestClassifier(
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
        class_weight="balanced",
        random_state=args.seed,
        n_jobs=-1,
    )
    classifier.fit(x_train, train_frame["label"])

    predictions = classifier.predict(x_test)
    probabilities = classifier.predict_proba(x_test)[:, 1]
    metrics = binary_metrics(test_frame["label"], predictions, probabilities)

    output_dir = Path(args.output_dir)
    artifacts = {
        "model": str(output_dir / "fake_review_rf.joblib"),
        "vectorizer": str(output_dir / "fake_review_vectorizer.joblib"),
        "feature_config": str(output_dir / "fake_review_feature_config.json"),
        "metadata": str(output_dir / "model_metadata.json"),
    }
    params = {
        "test_size": args.test_size,
        "seed": args.seed,
        "sample_size": args.sample_size,
        "max_features": args.max_features,
        "max_ngram": args.max_ngram,
        "min_df": args.min_df,
        "n_estimators": args.n_estimators,
        "min_samples_leaf": args.min_samples_leaf,
    }
    metadata = metadata_payload(
        model_name="fake_review_random_forest",
        model_version=args.model_version,
        dataset=dataset_info,
        metrics=metrics,
        params=params,
        artifacts=artifacts,
        limitations=[
            "Labels distinguish original reviews from generated fake reviews, not every real-world fraud pattern.",
            "Backend aggregation averages review-level probabilities at product level.",
        ],
    )

    if not args.no_save:
        dump_joblib(Path(artifacts["model"]), classifier)
        dump_joblib(Path(artifacts["vectorizer"]), vectorizer)
        write_json(
            Path(artifacts["feature_config"]),
            {
                "text_column": "text",
                "label_column": "label",
                "label_mapping": {"0": "original_authentic", "1": "computer_generated_fake"},
                "vectorizer": {
                    "type": "TfidfVectorizer",
                    "max_features": args.max_features,
                    "ngram_range": [1, args.max_ngram],
                    "min_df": args.min_df,
                    "sublinear_tf": True,
                },
            },
        )
        write_json(Path(artifacts["metadata"]), metadata)

    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--split", default="train")
    parser.add_argument("--local-csv")
    parser.add_argument("--output-dir", default=str(default_artifacts_dir()))
    parser.add_argument("--model-version", default="0.1.0")
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=30000)
    parser.add_argument("--max-ngram", type=int, default=2)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--no-save", action="store_true")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    result = train_fake_review_model(args)
    print(result["metrics"])
