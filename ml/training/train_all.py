"""Run the TrustScore training workflow from local ``ml/data`` samples."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

try:
    from ml.training.calibrate_trustscore import build_parser as build_calibrator_parser
    from ml.training.calibrate_trustscore import calibrate_trustscore
    from ml.training.common import default_artifacts_dir, write_json
    from ml.training.download_datasets import build_parser as build_download_parser
    from ml.training.download_datasets import default_data_dir, download_datasets
    from ml.training.train_fake_review_model import build_parser as build_fake_parser
    from ml.training.train_fake_review_model import train_fake_review_model
    from ml.training.train_risk_models import build_parser as build_risk_parser
    from ml.training.train_risk_models import train_risk_models
    from ml.training.train_sentiment_model import build_parser as build_sentiment_parser
    from ml.training.train_sentiment_model import train_sentiment_model
except ModuleNotFoundError:
    from calibrate_trustscore import build_parser as build_calibrator_parser  # type: ignore[no-redef]
    from calibrate_trustscore import calibrate_trustscore  # type: ignore[no-redef]
    from common import default_artifacts_dir, write_json  # type: ignore[no-redef]
    from download_datasets import build_parser as build_download_parser  # type: ignore[no-redef]
    from download_datasets import default_data_dir, download_datasets  # type: ignore[no-redef]
    from train_fake_review_model import build_parser as build_fake_parser  # type: ignore[no-redef]
    from train_fake_review_model import train_fake_review_model  # type: ignore[no-redef]
    from train_risk_models import build_parser as build_risk_parser  # type: ignore[no-redef]
    from train_risk_models import train_risk_models  # type: ignore[no-redef]
    from train_sentiment_model import build_parser as build_sentiment_parser  # type: ignore[no-redef]
    from train_sentiment_model import train_sentiment_model  # type: ignore[no-redef]


def train_all(args: argparse.Namespace) -> dict[str, Any]:
    """Run dataset download and all configured model training steps."""
    data_dir = Path(args.data_dir)
    artifacts_dir = Path(args.output_dir)
    if not args.skip_download:
        download_args = build_download_parser().parse_args(
            [
                "--output-dir",
                str(data_dir),
                "--seed",
                str(args.seed),
                "--fake-sample-size",
                str(args.fake_sample_size),
                "--amazon-review-sample-size",
                str(args.sentiment_sample_size),
                "--amazon-meta-sample-size",
                str(args.risk_sample_size),
                "--target-policy-sample-size",
                str(args.policy_sample_size),
                "--categories",
                *args.categories,
            ]
        )
        download_datasets(download_args)

    fake_metadata = train_fake_review_model(
        build_fake_parser().parse_args(
            [
                "--local-csv",
                str(data_dir / "fake_reviews.csv"),
                "--output-dir",
                str(artifacts_dir / "fake_review"),
                "--model-version",
                args.model_version,
                "--seed",
                str(args.seed),
                "--test-size",
                str(args.test_size),
                "--sample-size",
                str(args.fake_sample_size),
                "--n-estimators",
                str(args.fake_estimators),
                "--min-df",
                "1",
            ]
        )
    )

    sentiment_metadata = train_sentiment_model(
        build_sentiment_parser().parse_args(
            [
                "--local-csv",
                str(data_dir / "amazon_reviews_sample.csv"),
                "--output-dir",
                str(artifacts_dir / "sentiment"),
                "--model-version",
                args.model_version,
                "--seed",
                str(args.seed),
                "--test-size",
                str(args.test_size),
                "--sample-size",
                str(args.sentiment_sample_size),
                "--min-df",
                "1",
            ]
        )
    )

    risk_args = [
        "--amazon-meta-csv",
        str(data_dir / "amazon_meta_sample.csv"),
        "--output-dir",
        str(artifacts_dir / "risk"),
        "--model-version",
        args.model_version,
        "--seed",
        str(args.seed),
        "--test-size",
        str(args.test_size),
        "--sample-size",
        str(args.risk_sample_size),
        "--n-estimators",
        str(args.risk_estimators),
        "--min-df",
        "1",
    ]
    policy_csv = data_dir / "target_policy_sample.csv"
    if policy_csv.exists():
        risk_args.extend(["--policy-csv", str(policy_csv)])
    risk_metadata = train_risk_models(build_risk_parser().parse_args(risk_args))

    labels_csv = data_dir / "human_trustscore_labels.csv"
    calibrator_args = ["--output-dir", str(artifacts_dir), "--model-version", args.model_version]
    if labels_csv.exists():
        calibrator_args.extend(["--local-csv", str(labels_csv)])
    else:
        calibrator_args.append("--no-save")
    calibrator_metadata = calibrate_trustscore(
        build_calibrator_parser().parse_args(calibrator_args)
    )

    summary = {
        "artifacts_dir": str(artifacts_dir),
        "fake_review": fake_metadata,
        "sentiment": sentiment_metadata,
        "risk": risk_metadata,
        "trustscore_calibration": calibrator_metadata,
    }
    write_json(artifacts_dir / "training_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(default_data_dir()))
    parser.add_argument("--output-dir", default=str(default_artifacts_dir()))
    parser.add_argument("--model-version", default="0.1.0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--categories", nargs="+", default=["All_Beauty"])
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--fake-sample-size", type=int, default=20_000)
    parser.add_argument("--sentiment-sample-size", type=int, default=20_000)
    parser.add_argument("--risk-sample-size", type=int, default=20_000)
    parser.add_argument("--policy-sample-size", type=int, default=10_000)
    parser.add_argument("--fake-estimators", type=int, default=300)
    parser.add_argument("--risk-estimators", type=int, default=200)
    return parser


if __name__ == "__main__":
    result = train_all(build_parser().parse_args())
    print(result.keys())
