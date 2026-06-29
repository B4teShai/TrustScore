"""v2 sentiment model: multi-category data, word+char TF-IDF, model selection.

v1: single-category, word-only TF-IDF + LogReg (acc 0.864). v2 trains on the multi-category
download, adds char n-grams, and picks the better of LogReg / calibrated LinearSVC by macro-F1.
Saved as a single Pipeline that ``predict_proba(list[str])`` with classes exactly
``negative / neutral / positive`` — drop-in for ``sentiment_service.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

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
except ModuleNotFoundError:  # pragma: no cover
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


DATASET_URL = "https://amazon-reviews-2023.github.io/main.html"


def _default_data() -> Path:
    return Path(__file__).resolve().parents[2] / "data"


def sentiment_label_from_rating(value: object) -> str | None:
    rating = parse_float(value, default=-1)
    if rating < 1:
        return None
    if rating <= 2:
        return "negative"
    if rating < 4:
        return "neutral"
    return "positive"


def _vectorizer(args: argparse.Namespace) -> FeatureUnion:
    word = TfidfVectorizer(
        max_features=args.max_features, ngram_range=(1, args.word_ngram), min_df=2,
        sublinear_tf=True, strip_accents="unicode",
    )
    char = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(args.char_min, args.char_max), max_features=args.max_features // 2,
        min_df=2, sublinear_tf=True, strip_accents="unicode",
    )
    return FeatureUnion([("word", word), ("char", char)])


def load_frame(args: argparse.Namespace) -> tuple[pd.DataFrame, DatasetInfo]:
    csv_path = args.local_csv or str(_default_data() / "amazon_reviews_sample.csv")
    frame = load_local_csv(csv_path)
    require_columns(frame, ("rating", "text"))
    frame = frame.copy()
    title = frame["title"].fillna("").astype(str) if "title" in frame.columns else pd.Series([""] * len(frame))
    frame["text"] = (title + " " + frame["text"].fillna("").astype(str)).map(clean_text)
    frame["label"] = frame["rating"].map(sentiment_label_from_rating)
    frame = frame[(frame["text"] != "") & frame["label"].notna()].reset_index(drop=True)
    return frame, DatasetInfo(
        name=Path(csv_path).name, source=str(Path(csv_path)), url=DATASET_URL, row_count=len(frame)
    )


def train(args: argparse.Namespace) -> dict[str, Any]:
    frame, dataset_info = load_frame(args)
    print(f"[sentiment-v2] {len(frame)} rows; classes={frame['label'].value_counts().to_dict()}")
    train_frame, test_frame = safe_train_test_split(frame, "label", args.test_size, args.seed)
    tr_frame, val_frame = train_test_split(
        train_frame, test_size=0.2, random_state=args.seed, stratify=train_frame["label"]
    )

    candidates = {
        "logreg": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=args.seed),
        "linear_svc": CalibratedClassifierCV(LinearSVC(class_weight="balanced", random_state=args.seed), cv=3),
    }
    selection: dict[str, Any] = {}
    best_name, best_f1, best_clf = None, -1.0, None
    for name, clf in candidates.items():
        pipe = Pipeline([("vectorizer", _vectorizer(args)), ("classifier", clf)])
        pipe.fit(tr_frame["text"], tr_frame["label"])
        m = multiclass_metrics(val_frame["label"], pipe.predict(val_frame["text"]))
        selection[name] = m
        print(f"[sentiment-v2] {name:12} val acc={m['accuracy']} f1_macro={m['f1_macro']}")
        if m["f1_macro"] > best_f1:
            best_name, best_f1, best_clf = name, m["f1_macro"], clf

    pipeline = Pipeline([("vectorizer", _vectorizer(args)), ("classifier", best_clf)])
    pipeline.fit(train_frame["text"], train_frame["label"])
    test_metrics = multiclass_metrics(test_frame["label"], pipeline.predict(test_frame["text"]))
    print(f"[sentiment-v2] WINNER={best_name} test={test_metrics}")

    output_dir = Path(args.output_dir)
    artifacts = {
        "pipeline": str(output_dir / "sentiment_tfidf_logreg.joblib"),
        "metadata": str(output_dir / "sentiment_model_metadata.json"),
    }
    metadata = metadata_payload(
        model_name=f"sentiment_v2_{best_name}",
        model_version=args.model_version,
        dataset=dataset_info,
        metrics={"test": test_metrics, "validation_model_selection": selection, "winner": best_name},
        params={
            "test_size": args.test_size, "seed": args.seed, "max_features": args.max_features,
            "word_ngram": [1, args.word_ngram], "char_ngram": [args.char_min, args.char_max],
        },
        artifacts=artifacts,
        limitations=[
            "Sentiment labels are weak labels derived from star ratings.",
            "Neutral (3-star) is the hardest class and drives most macro-F1 loss.",
        ],
    )
    if not args.no_save:
        dump_joblib(Path(artifacts["pipeline"]), pipeline)
        write_json(Path(artifacts["metadata"]), metadata)
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-csv")
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parents[2] / "artifacts" / "v2" / "sentiment"))
    parser.add_argument("--model-version", default="0.2.0")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=50000)
    parser.add_argument("--word-ngram", type=int, default=2)
    parser.add_argument("--char-min", type=int, default=3)
    parser.add_argument("--char-max", type=int, default=5)
    parser.add_argument("--no-save", action="store_true")
    return parser


if __name__ == "__main__":
    result = train(build_parser().parse_args())
    print(result["metrics"]["test"])
