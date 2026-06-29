"""v2 fake-review classifier: word+char TF-IDF, model selection across estimators.

v1 used word-only TF-IDF + RandomForest (acc 0.896, ROC-AUC 0.961). v2 adds char_wb
n-grams (which catch machine-generated surface artifacts), benchmarks several classifiers
on a validation split, and keeps the best by ROC-AUC. The saved artifacts stay a
**vectorizer + binary classifier pair** (fake = last column) so the backend's
``fake_review_service.py`` loads them unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC

try:
    from ml.training.common import (
        DatasetInfo,
        binary_metrics,
        clean_text,
        dump_joblib,
        load_local_csv,
        metadata_payload,
        require_columns,
        safe_train_test_split,
        write_json,
    )
    from ml.training.v2.features import build_fake_review_vectorizer
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from common import (  # type: ignore[no-redef]
        DatasetInfo,
        binary_metrics,
        clean_text,
        dump_joblib,
        load_local_csv,
        metadata_payload,
        require_columns,
        safe_train_test_split,
        write_json,
    )
    from v2.features import build_fake_review_vectorizer  # type: ignore[no-redef]


DATASET_URL = "https://huggingface.co/datasets/theArijitDas/Fake-Reviews-Dataset"


def _default_data() -> Path:
    return Path(__file__).resolve().parents[2] / "data"


def _candidates(args: argparse.Namespace) -> dict[str, Any]:
    """Binary classifiers exposing predict_proba (calibrate LinearSVC, which does not)."""
    return {
        "logreg": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=args.seed),
        "linear_svc": CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", random_state=args.seed), cv=3
        ),
        "complement_nb": ComplementNB(),
        "random_forest": RandomForestClassifier(
            n_estimators=args.n_estimators,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=args.seed,
            n_jobs=-1,
        ),
    }


def load_dataset_frame(args: argparse.Namespace) -> tuple[pd.DataFrame, DatasetInfo]:
    csv_path = args.local_csv or str(_default_data() / "fake_reviews.csv")
    frame = load_local_csv(csv_path)
    require_columns(frame, ("text", "label"))
    frame = frame.copy()
    frame["text"] = frame["text"].map(clean_text)
    frame["label"] = frame["label"].astype(int)
    frame = frame[(frame["text"] != "") & frame["label"].isin([0, 1])].reset_index(drop=True)
    return frame, DatasetInfo(
        name=Path(csv_path).name, source=str(Path(csv_path)), url=DATASET_URL, row_count=len(frame)
    )


def train(args: argparse.Namespace) -> dict[str, Any]:
    frame, dataset_info = load_dataset_frame(args)
    print(f"[fake-v2] {len(frame)} rows from {dataset_info.source}")

    # train / val / test = 64 / 16 / 20
    train_frame, test_frame = safe_train_test_split(frame, "label", args.test_size, args.seed)
    tr_frame, val_frame = train_test_split(
        train_frame, test_size=0.2, random_state=args.seed, stratify=train_frame["label"]
    )

    vectorizer = build_fake_review_vectorizer(
        max_features=args.max_features,
        word_ngram=args.word_ngram,
        char_min=args.char_min,
        char_max=args.char_max,
    )
    x_tr = vectorizer.fit_transform(tr_frame["text"])
    x_val = vectorizer.transform(val_frame["text"])
    x_test = vectorizer.transform(test_frame["text"])

    selection: dict[str, dict[str, Any]] = {}
    best_name, best_auc, best_model = None, -1.0, None
    for name, model in _candidates(args).items():
        model.fit(x_tr, tr_frame["label"])
        val_prob = model.predict_proba(x_val)[:, -1]
        val_pred = model.predict(x_val)
        m = binary_metrics(val_frame["label"], val_pred, val_prob)
        selection[name] = m
        score = m.get("roc_auc", m["f1"])
        print(f"[fake-v2] {name:14} val acc={m['accuracy']} f1={m['f1']} auc={m.get('roc_auc')}")
        if score > best_auc:
            best_name, best_auc, best_model = name, score, model

    # refit winner on train+val, evaluate on held-out test
    x_fit = vectorizer.transform(train_frame["text"])
    best_model.fit(x_fit, train_frame["label"])
    test_prob = best_model.predict_proba(x_test)[:, -1]
    test_pred = best_model.predict(x_test)
    test_metrics = binary_metrics(test_frame["label"], test_pred, test_prob)
    print(f"[fake-v2] WINNER={best_name} test={test_metrics}")

    output_dir = Path(args.output_dir)
    artifacts = {
        "model": str(output_dir / "fake_review_rf.joblib"),
        "vectorizer": str(output_dir / "fake_review_vectorizer.joblib"),
        "feature_config": str(output_dir / "fake_review_feature_config.json"),
        "metadata": str(output_dir / "model_metadata.json"),
    }
    metadata = metadata_payload(
        model_name=f"fake_review_v2_{best_name}",
        model_version=args.model_version,
        dataset=dataset_info,
        metrics={"test": test_metrics, "validation_model_selection": selection, "winner": best_name},
        params={
            "test_size": args.test_size,
            "seed": args.seed,
            "max_features": args.max_features,
            "word_ngram": args.word_ngram,
            "char_ngram": [args.char_min, args.char_max],
            "candidates": list(_candidates(args).keys()),
        },
        artifacts=artifacts,
        limitations=[
            "Labels distinguish original from generated reviews, not every real-world fraud pattern.",
            "Backend averages review-level fake probabilities at product level.",
        ],
    )

    if not args.no_save:
        dump_joblib(Path(artifacts["model"]), best_model)
        dump_joblib(Path(artifacts["vectorizer"]), vectorizer)
        write_json(
            Path(artifacts["feature_config"]),
            {
                "text_column": "text",
                "label_column": "label",
                "label_mapping": {"0": "original_authentic", "1": "computer_generated_fake"},
                "vectorizer": {
                    "type": "FeatureUnion(word_tfidf + char_wb_tfidf)",
                    "winner": best_name,
                    "max_features": args.max_features,
                    "word_ngram": [1, args.word_ngram],
                    "char_ngram": [args.char_min, args.char_max],
                },
            },
        )
        write_json(Path(artifacts["metadata"]), metadata)
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-csv")
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parents[2] / "artifacts" / "v2"))
    parser.add_argument("--model-version", default="0.2.0")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=40000)
    parser.add_argument("--word-ngram", type=int, default=2)
    parser.add_argument("--char-min", type=int, default=3)
    parser.add_argument("--char-max", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--no-save", action="store_true")
    return parser


if __name__ == "__main__":
    result = train(build_parser().parse_args())
    print(result["metrics"]["test"])
