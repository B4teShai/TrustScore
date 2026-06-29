"""v2 risk models: numeric + text features, with a text-only ablation.

Seller/price/policy weak labels are functions of numeric metadata and policy-text flags.
v1 used text only and scored ~0.58/0.60. v2 adds the numeric features the labels are
derived from via a ColumnTransformer and reports the text-only-vs-+numeric ablation as the
key result. Artifacts (DataFrame-consuming Pipelines + feature_spec.json) stay drop-in for
the FastAPI backend's v2 risk path.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

try:
    from ml.training.common import (
        DatasetInfo,
        clean_text,
        dump_joblib,
        load_local_csv,
        metadata_payload,
        multiclass_metrics,
        parse_float,
        require_columns,
        safe_train_test_split,
        write_json,
    )
    from ml.training.train_risk_models import policy_label, price_label, seller_label
    from ml.training.v2.features import (
        RISK_FEATURE_SPECS,
        log_review_count,
        make_risk_pipeline,
        make_text_only_pipeline,
        policy_flags,
        price_ratio,
    )
except ModuleNotFoundError:  # pragma: no cover - script execution fallback
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from common import (  # type: ignore[no-redef]
        DatasetInfo,
        clean_text,
        dump_joblib,
        load_local_csv,
        metadata_payload,
        multiclass_metrics,
        parse_float,
        require_columns,
        safe_train_test_split,
        write_json,
    )
    from train_risk_models import policy_label, price_label, seller_label  # type: ignore[no-redef]
    from v2.features import (  # type: ignore[no-redef]
        RISK_FEATURE_SPECS,
        log_review_count,
        make_risk_pipeline,
        make_text_only_pipeline,
        policy_flags,
        price_ratio,
    )


AMAZON_DATASET_URL = "https://amazon-reviews-2023.github.io/main.html"
TARGET_DATASET_URL = "https://github.com/luminati-io/Target-dataset-samples"


def _default_data() -> Path:
    return Path(__file__).resolve().parents[2] / "data"


def _classifier(args: argparse.Namespace) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
        class_weight="balanced",
        random_state=args.seed,
        n_jobs=-1,
    )


def _fit_and_eval(
    frame: pd.DataFrame,
    *,
    numeric_features: list[str],
    text_feature: str,
    label_column: str,
    args: argparse.Namespace,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Return (numeric+text pipeline, +numeric metrics, text-only ablation metrics)."""
    train_frame, test_frame = safe_train_test_split(frame, label_column, args.test_size, args.seed)

    full = make_risk_pipeline(
        numeric_features=numeric_features,
        text_feature=text_feature,
        classifier=_classifier(args),
        max_features=args.max_features,
        max_ngram=args.max_ngram,
        min_df=args.min_df,
    )
    full.fit(train_frame, train_frame[label_column])
    full_metrics = multiclass_metrics(test_frame[label_column], full.predict(test_frame))

    text_only = make_text_only_pipeline(
        text_feature=text_feature,
        classifier=_classifier(args),
        max_features=args.max_features,
        max_ngram=args.max_ngram,
        min_df=args.min_df,
    )
    text_only.fit(train_frame, train_frame[label_column])
    text_metrics = multiclass_metrics(test_frame[label_column], text_only.predict(test_frame))
    return full, full_metrics, text_metrics


def load_meta(args: argparse.Namespace) -> tuple[pd.DataFrame, DatasetInfo]:
    csv_path = args.amazon_meta_csv or str(_default_data() / "amazon_meta_sample.csv")
    frame = load_local_csv(csv_path)
    require_columns(frame, ("title", "average_rating", "rating_number", "price"))
    frame = frame.copy()
    frame["main_category"] = frame.get("main_category", frame.get("source_category", "unknown"))
    frame["price_float"] = frame["price"].map(parse_float)
    return frame, DatasetInfo(
        name=Path(csv_path).name, source=str(Path(csv_path)), url=AMAZON_DATASET_URL, row_count=len(frame)
    )


def build_risk_frames(meta: pd.DataFrame) -> pd.DataFrame:
    meta = meta.copy()
    meta["seller_label"] = meta.apply(seller_label, axis=1)
    medians = (
        meta[meta["price_float"] > 0].groupby("main_category")["price_float"].median().to_dict()
    )
    meta["category_median"] = meta["main_category"].map(medians)
    meta["price_label"] = meta.apply(
        lambda r: price_label(r["price_float"], medians.get(r["main_category"], float(np.nan))), axis=1
    )
    store = (
        meta["store"].fillna("").astype(str)
        if "store" in meta.columns
        else pd.Series([""] * len(meta))
    )
    meta["seller_text"] = (
        store + " " + meta["title"].fillna("").astype(str) + " " + meta["main_category"].fillna("").astype(str)
    ).map(clean_text)
    meta["price_text"] = (
        meta["title"].fillna("").astype(str) + " " + meta["main_category"].fillna("").astype(str)
    ).map(clean_text)
    # numeric features (names match RISK_FEATURE_SPECS / backend reconstruction)
    meta["rating"] = meta["average_rating"].map(lambda v: parse_float(v, 0.0))
    meta["log_review_count"] = meta["rating_number"].map(log_review_count)
    meta["price"] = meta["price_float"]
    meta["price_ratio"] = meta.apply(
        lambda r: price_ratio(r["price_float"], r.get("category_median")), axis=1
    )
    return meta


def load_policy_frame(args: argparse.Namespace) -> pd.DataFrame | None:
    csv_path = args.policy_csv or str(_default_data() / "target_policy_sample.csv")
    path = Path(csv_path)
    if not path.exists():
        return None
    frame = load_local_csv(path)
    column = args.policy_column if args.policy_column in frame.columns else None
    if column is None:
        # fall back to the first text-ish column that mentions policy/return/shipping
        candidates = [c for c in frame.columns if any(k in c.lower() for k in ("polic", "return", "shipping"))]
        if not candidates:
            return None
        column = candidates[0]
    frame = frame.copy()
    frame["policy_text"] = frame[column].map(clean_text)
    frame["policy_label"] = frame[column].map(policy_label)
    flags = frame[column].map(policy_flags).apply(pd.Series)
    frame = pd.concat([frame, flags], axis=1)
    frame = frame[frame["policy_text"] != ""].reset_index(drop=True)
    return frame


def train(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta, meta_info = load_meta(args)
    print(f"[risk-v2] loaded {len(meta)} metadata rows from {meta_info.source}")
    meta = build_risk_frames(meta)

    results: dict[str, Any] = {}
    saved: dict[str, str] = {}

    # ---- seller reliability ----
    spec = RISK_FEATURE_SPECS["seller_reliability"]
    seller_frame = meta[meta["seller_text"] != ""]
    seller_model, seller_full, seller_text = _fit_and_eval(
        seller_frame,
        numeric_features=spec["numeric_features"],
        text_feature=spec["text_feature"],
        label_column="seller_label",
        args=args,
    )
    results["seller_reliability"] = {
        "numeric_plus_text": seller_full,
        "text_only_ablation": seller_text,
        "n_rows": int(len(seller_frame)),
    }
    saved.update(_save_model(output_dir, "seller_reliability", "seller_reliability_tfidf_rf", seller_model, spec, args))

    # ---- price safety ----
    spec = RISK_FEATURE_SPECS["price_safety"]
    price_frame = meta[(meta["price_text"] != "") & (meta["price_label"] != "unknown")]
    price_model, price_full, price_text = _fit_and_eval(
        price_frame,
        numeric_features=spec["numeric_features"],
        text_feature=spec["text_feature"],
        label_column="price_label",
        args=args,
    )
    results["price_safety"] = {
        "numeric_plus_text": price_full,
        "text_only_ablation": price_text,
        "n_rows": int(len(price_frame)),
    }
    saved.update(_save_model(output_dir, "price_safety", "price_safety_tfidf_rf", price_model, spec, args))

    # ---- return policy clarity (v1 skipped this entirely) ----
    policy_frame = load_policy_frame(args)
    if policy_frame is not None and len(policy_frame) >= 8:
        spec = RISK_FEATURE_SPECS["return_policy_clarity"]
        policy_model, policy_full, policy_text = _fit_and_eval(
            policy_frame,
            numeric_features=spec["numeric_features"],
            text_feature=spec["text_feature"],
            label_column="policy_label",
            args=args,
        )
        results["return_policy_clarity"] = {
            "numeric_plus_text": policy_full,
            "text_only_ablation": policy_text,
            "n_rows": int(len(policy_frame)),
        }
        saved.update(
            _save_model(output_dir, "return_policy_clarity", "policy_clarity_tfidf_rf", policy_model, spec, args)
        )
    else:
        results["return_policy_clarity"] = {"status": "skipped_missing_policy_csv"}

    metadata = metadata_payload(
        model_name="trustscore_v2_numeric_text_risk_models",
        model_version=args.model_version,
        dataset=meta_info,
        metrics=results,
        params={
            "test_size": args.test_size,
            "seed": args.seed,
            "max_features": args.max_features,
            "max_ngram": args.max_ngram,
            "min_df": args.min_df,
            "n_estimators": args.n_estimators,
            "classifier": "RandomForestClassifier",
        },
        artifacts=saved,
        limitations=[
            "Seller/price/policy labels remain weak labels derived from public metadata and "
            "policy-text heuristics; numeric features partly reconstruct the labelling rule, so "
            "held-out accuracy is high by construction. The ablation isolates the genuine lift "
            "over v1's text-only baseline.",
        ],
    )
    if not args.no_save:
        write_json(output_dir / "risk_model_metadata.json", metadata)
    return metadata


def _save_model(
    output_dir: Path, model_key: str, filename: str, model: Any, spec: dict[str, Any], args: argparse.Namespace
) -> dict[str, str]:
    if args.no_save:
        return {}
    artifact_path = output_dir / f"{filename}.joblib"
    spec_path = output_dir / f"{filename}_feature_spec.json"
    dump_joblib(artifact_path, model)
    write_json(
        spec_path,
        {
            "model": model_key,
            "artifact": str(artifact_path),
            "input": "pandas.DataFrame",
            "numeric_features": spec["numeric_features"],
            "text_feature": spec["text_feature"],
            "classes": [str(c) for c in model.classes_],
            "label_score_map": spec["label_score_map"],
        },
    )
    return {model_key: str(artifact_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amazon-meta-csv")
    parser.add_argument("--policy-csv")
    parser.add_argument("--policy-column", default="shipping_returns_policy")
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parents[2] / "artifacts" / "v2" / "risk"))
    parser.add_argument("--model-version", default="0.2.0")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=5000)
    parser.add_argument("--max-ngram", type=int, default=2)
    parser.add_argument("--min-df", type=int, default=3)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--no-save", action="store_true")
    return parser


if __name__ == "__main__":
    result = train(build_parser().parse_args())
    print(result["metrics"])
