"""Optional calibration for a human-labeled TrustScore dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

try:
    from ml.training.common import (
        DatasetInfo,
        default_artifacts_dir,
        dump_joblib,
        load_local_csv,
        metadata_payload,
        require_columns,
        safe_train_test_split,
        write_json,
    )
except ModuleNotFoundError:
    from common import (  # type: ignore[no-redef]
        DatasetInfo,
        default_artifacts_dir,
        dump_joblib,
        load_local_csv,
        metadata_payload,
        require_columns,
        safe_train_test_split,
        write_json,
    )


FEATURE_COLUMNS = (
    "review_authenticity",
    "seller_reliability",
    "sentiment",
    "return_policy_clarity",
    "price_safety",
    "user_feedback_history",
)


def calibrate_trustscore(args: argparse.Namespace) -> dict[str, Any]:
    if not args.local_csv:
        return {
            "status": "skipped",
            "reason": "No human-labeled TrustScore CSV was provided.",
            "required_columns": [*FEATURE_COLUMNS, "trust_score"],
        }

    frame = load_local_csv(args.local_csv)
    require_columns(frame, (*FEATURE_COLUMNS, "trust_score"))
    train_frame, test_frame = safe_train_test_split(frame, "trust_score", args.test_size, args.seed)
    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.seed,
        n_jobs=-1,
    )
    model.fit(train_frame[list(FEATURE_COLUMNS)], train_frame["trust_score"])
    predictions = model.predict(test_frame[list(FEATURE_COLUMNS)])
    metrics = {
        "mae": round(float(mean_absolute_error(test_frame["trust_score"], predictions)), 4),
        "r2": round(float(r2_score(test_frame["trust_score"], predictions)), 4),
    }
    output_dir = Path(args.output_dir)
    artifacts = {
        "calibrator": str(output_dir / "trustscore_calibrator_rf.joblib"),
        "metadata": str(output_dir / "trustscore_calibrator_metadata.json"),
    }
    metadata = metadata_payload(
        model_name="trustscore_human_label_calibrator",
        model_version=args.model_version,
        dataset=DatasetInfo(
            name=Path(args.local_csv).name,
            source=str(Path(args.local_csv)),
            url=None,
            row_count=len(frame),
        ),
        metrics=metrics,
        params={
            "test_size": args.test_size,
            "seed": args.seed,
            "n_estimators": args.n_estimators,
            "min_samples_leaf": args.min_samples_leaf,
            "features": list(FEATURE_COLUMNS),
        },
        artifacts=artifacts,
        limitations=[
            "Requires human-labeled TrustScore examples; public datasets do not provide this target.",
        ],
    )
    if not args.no_save:
        dump_joblib(Path(artifacts["calibrator"]), model)
        write_json(Path(artifacts["metadata"]), metadata)
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-csv")
    parser.add_argument("--output-dir", default=str(default_artifacts_dir()))
    parser.add_argument("--model-version", default="0.1.0")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--no-save", action="store_true")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    result = calibrate_trustscore(args)
    print(result)
