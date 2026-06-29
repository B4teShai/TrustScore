"""Smoke tests for the v2 training package on tiny local CSVs.

Confirms each v2 trainer writes backend-compatible artifacts to a v2 output dir, that the
risk pipelines consume a DataFrame and expose string classes, and that feature_spec.json
is emitted. Mirrors the structure of test_training_scripts.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from ml.training.v2.train_fake_review_model_v2 import build_parser as fake_parser
from ml.training.v2.train_fake_review_model_v2 import train as train_fake_v2
from ml.training.v2.train_risk_models_v2 import build_parser as risk_parser
from ml.training.v2.train_risk_models_v2 import train as train_risk_v2
from ml.training.v2.train_sentiment_model_v2 import build_parser as sentiment_parser
from ml.training.v2.train_sentiment_model_v2 import train as train_sentiment_v2


def _csv(path: Path, rows: list[dict[str, object]]) -> Path:
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_fake_v2_writes_vectorizer_and_binary_model(tmp_path: Path) -> None:
    rows = []
    for _ in range(12):
        rows.append({"text": "genuine honest review about real product use over time", "label": 0})
        rows.append({"text": "perfect perfect amazing wow buy now best ever fake", "label": 1})
    csv = _csv(tmp_path / "fake.csv", rows)
    out = tmp_path / "v2"
    args = fake_parser().parse_args(
        ["--local-csv", str(csv), "--output-dir", str(out), "--max-features", "200", "--char-max", "4"]
    )
    meta = train_fake_v2(args)

    model = joblib.load(out / "fake_review_rf.joblib")
    vectorizer = joblib.load(out / "fake_review_vectorizer.joblib")
    proba = model.predict_proba(vectorizer.transform(["totally real review"]))
    assert proba.shape[1] == 2  # binary, fake = last column (backend contract)
    assert "winner" in meta["metrics"]
    assert (out / "model_metadata.json").exists()


def test_sentiment_v2_keeps_three_string_classes(tmp_path: Path) -> None:
    rows = []
    for _ in range(8):
        rows.append({"rating": 5, "title": "love", "text": "excellent fantastic happy great love it"})
        rows.append({"rating": 3, "title": "ok", "text": "okay average neutral fine acceptable middling"})
        rows.append({"rating": 1, "title": "bad", "text": "terrible awful broken hate disappointing poor"})
    csv = _csv(tmp_path / "rev.csv", rows)
    out = tmp_path / "v2" / "sentiment"
    args = sentiment_parser().parse_args(
        ["--local-csv", str(csv), "--output-dir", str(out), "--max-features", "200", "--char-max", "4"]
    )
    train_sentiment_v2(args)

    pipeline = joblib.load(out / "sentiment_tfidf_logreg.joblib")
    assert set(str(c) for c in pipeline.classes_) <= {"negative", "neutral", "positive"}
    pred = pipeline.predict(["excellent fantastic love"])
    assert str(pred[0]) in {"negative", "neutral", "positive"}


def test_risk_v2_builds_dataframe_pipeline_with_feature_spec(tmp_path: Path) -> None:
    meta_rows = []
    for i in range(30):
        meta_rows.append(
            {"title": f"premium beauty serum {i}", "average_rating": 4.6, "rating_number": 800,
             "price": 25.0, "store": "TrustedBeauty", "main_category": "All_Beauty"}
        )
        meta_rows.append(
            {"title": f"generic item {i}", "average_rating": 3.0, "rating_number": 3,
             "price": 90.0, "store": "Unknown", "main_category": "All_Beauty"}
        )
    meta_csv = _csv(tmp_path / "meta.csv", meta_rows)
    policy_rows = [
        {"shipping_returns_policy": "30-day return and refund with warranty exchange available"} for _ in range(8)
    ] + [{"shipping_returns_policy": "no returns"} for _ in range(8)]
    policy_csv = _csv(tmp_path / "policy.csv", policy_rows)

    out = tmp_path / "v2" / "risk"
    args = risk_parser().parse_args(
        ["--amazon-meta-csv", str(meta_csv), "--policy-csv", str(policy_csv),
         "--output-dir", str(out), "--max-features", "200", "--min-df", "1", "--n-estimators", "20"]
    )
    meta = train_risk_v2(args)

    spec = json.loads((out / "seller_reliability_tfidf_rf_feature_spec.json").read_text())
    assert spec["numeric_features"] == ["rating", "log_review_count"]
    assert spec["text_feature"] == "seller_text"

    model = joblib.load(out / "seller_reliability_tfidf_rf.joblib")
    frame = pd.DataFrame([{"rating": 4.6, "log_review_count": 6.5, "seller_text": "premium beauty serum"}])
    assert model.predict_proba(frame).shape[1] == len(model.classes_)

    # the ablation must be recorded (text-only vs +numeric) — the headline result
    seller = meta["metrics"]["seller_reliability"]
    assert "numeric_plus_text" in seller and "text_only_ablation" in seller
