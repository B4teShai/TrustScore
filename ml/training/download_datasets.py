"""Download sampled TrustScore training datasets into ``ml/data``."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from ml.training.common import load_hf_data_files, now_iso, repo_root, sample_frame, write_json
    from ml.training.train_risk_models import amazon_meta_files
    from ml.training.train_sentiment_model import amazon_review_file
except ModuleNotFoundError:
    from common import load_hf_data_files, now_iso, repo_root, sample_frame, write_json  # type: ignore[no-redef]
    from train_risk_models import amazon_meta_files  # type: ignore[no-redef]
    from train_sentiment_model import amazon_review_file  # type: ignore[no-redef]


FAKE_REVIEW_DATASET = "theArijitDas/Fake-Reviews-Dataset"
TARGET_POLICY_URL = (
    "https://raw.githubusercontent.com/luminati-io/"
    "Target-dataset-samples/main/target-products.csv"
)


def default_data_dir() -> Path:
    return repo_root() / "ml" / "data"


def download_datasets(args: argparse.Namespace) -> dict[str, Any]:
    """Download bounded samples and return the written manifest."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []

    fake_frame = _load_fake_reviews(args)
    entries.append(
        _write_dataset(
            fake_frame,
            output_dir / "fake_reviews.csv",
            source=FAKE_REVIEW_DATASET,
            sample_size=args.fake_sample_size,
        )
    )

    review_frames = []
    meta_frames = []
    for category in args.categories:
        review_frames.append(
            load_hf_data_files(
                "json",
                [amazon_review_file(category)],
                split=args.amazon_split,
                sample_size=args.amazon_review_sample_size,
                seed=args.seed,
            )
        )
        meta_frame = load_hf_data_files(
            "parquet",
            [amazon_meta_files(category)],
            split=args.amazon_split,
            sample_size=args.amazon_meta_sample_size,
            seed=args.seed,
        )
        meta_frame["source_category"] = category
        meta_frames.append(meta_frame)

    entries.append(
        _write_dataset(
            pd.concat(review_frames, ignore_index=True),
            output_dir / "amazon_reviews_sample.csv",
            source="McAuley-Lab/Amazon-Reviews-2023:reviews",
            sample_size=args.amazon_review_sample_size,
            categories=args.categories,
        )
    )
    entries.append(
        _write_dataset(
            pd.concat(meta_frames, ignore_index=True),
            output_dir / "amazon_meta_sample.csv",
            source="McAuley-Lab/Amazon-Reviews-2023:metadata",
            sample_size=args.amazon_meta_sample_size,
            categories=args.categories,
        )
    )

    if not args.skip_target_policy:
        policy_frame = pd.read_csv(args.target_policy_url)
        policy_frame = sample_frame(policy_frame, args.target_policy_sample_size, args.seed)
        entries.append(
            _write_dataset(
                policy_frame,
                output_dir / "target_policy_sample.csv",
                source=args.target_policy_url,
                sample_size=args.target_policy_sample_size,
            )
        )

    if args.kaggle_deceptive_csv:
        kaggle_frame = pd.read_csv(args.kaggle_deceptive_csv)
        kaggle_frame = sample_frame(kaggle_frame, args.kaggle_sample_size, args.seed)
        entries.append(
            _write_dataset(
                kaggle_frame,
                output_dir / "kaggle_deceptive_opinion_sample.csv",
                source=str(args.kaggle_deceptive_csv),
                sample_size=args.kaggle_sample_size,
            )
        )

    manifest = {
        "created_at": now_iso(),
        "output_dir": str(output_dir),
        "seed": args.seed,
        "datasets": entries,
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def _load_fake_reviews(args: argparse.Namespace) -> pd.DataFrame:
    from datasets import load_dataset

    dataset = load_dataset(FAKE_REVIEW_DATASET, split=args.fake_split)
    frame = dataset.to_pandas()
    return sample_frame(frame, args.fake_sample_size, args.seed)


def _write_dataset(
    frame: pd.DataFrame,
    path: Path,
    *,
    source: str,
    sample_size: int | None,
    categories: list[str] | None = None,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return {
        "source": source,
        "categories": categories or [],
        "sample_size": sample_size,
        "row_count": len(frame),
        "path": str(path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(default_data_dir()))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--categories", nargs="+", default=["All_Beauty"])
    parser.add_argument("--fake-split", default="train")
    parser.add_argument("--fake-sample-size", type=int, default=20_000)
    parser.add_argument("--amazon-split", default="full")
    parser.add_argument("--amazon-review-sample-size", type=int, default=20_000)
    parser.add_argument("--amazon-meta-sample-size", type=int, default=20_000)
    parser.add_argument("--target-policy-url", default=TARGET_POLICY_URL)
    parser.add_argument("--target-policy-sample-size", type=int, default=10_000)
    parser.add_argument("--skip-target-policy", action="store_true")
    parser.add_argument("--kaggle-deceptive-csv")
    parser.add_argument("--kaggle-sample-size", type=int, default=10_000)
    return parser


if __name__ == "__main__":
    result = download_datasets(build_parser().parse_args())
    print(result)

