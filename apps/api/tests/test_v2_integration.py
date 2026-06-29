"""Backend integration tests for the v2 risk-model path.

Verifies the v2 DataFrame-consuming ColumnTransformer pipeline scores through
RiskModelService, that the v1 text-only artifact path still works, and that a missing
artifact degrades to rule scoring. Models are built inline so the test is self-contained
and does not depend on a prior training run.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.ml.risk_service import RiskModelService
from app.schemas.product_analysis import ProductPageData, SellerInfo


def _make_v2_seller_model(tmp_path: Path) -> Path:
    """A numeric+text seller model + sibling feature_spec.json (the v2 artifact shape)."""
    rows, labels = [], []
    for _ in range(20):
        rows.append({"rating": 4.8, "log_review_count": 6.5, "seller_text": "trusted beauty store premium"})
        labels.append("reliable")
        rows.append({"rating": 2.5, "log_review_count": 0.7, "seller_text": "unknown shop generic"})
        labels.append("weak")
    frame = pd.DataFrame(rows)

    transformer = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), ["rating", "log_review_count"]),
            ("txt", TfidfVectorizer(min_df=1), "seller_text"),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    pipeline = Pipeline([("features", transformer), ("classifier", RandomForestClassifier(n_estimators=25, random_state=0))])
    pipeline.fit(frame, labels)

    model_path = tmp_path / "seller_reliability_tfidf_rf.joblib"
    joblib.dump(pipeline, model_path)
    spec_path = tmp_path / "seller_reliability_tfidf_rf_feature_spec.json"
    spec_path.write_text(
        '{"model": "seller_reliability", "numeric_features": ["rating", "log_review_count"], '
        '"text_feature": "seller_text", "classes": ["reliable", "weak"], '
        '"label_score_map": {"reliable": 88, "weak": 32}}',
        encoding="utf-8",
    )
    return model_path


def _make_v1_seller_model(tmp_path: Path) -> Path:
    """A v1-style text-only pipeline (no feature_spec.json sibling)."""
    texts = ["trusted beauty store premium"] * 20 + ["unknown shop generic"] * 20
    labels = ["reliable"] * 20 + ["weak"] * 20
    pipeline = Pipeline(
        [("vectorizer", TfidfVectorizer(min_df=1)), ("classifier", RandomForestClassifier(n_estimators=25, random_state=0))]
    )
    pipeline.fit(texts, labels)
    model_path = tmp_path / "seller_v1.joblib"
    joblib.dump(pipeline, model_path)
    return model_path


def test_v2_seller_model_scores_via_dataframe_path(tmp_path: Path) -> None:
    model_path = _make_v2_seller_model(tmp_path)
    service = RiskModelService(seller_model_path=str(model_path), price_model_path=None, policy_model_path=None)

    reliable = service.score_product(
        ProductPageData(
            url="https://example.com/p1",
            product_title="Premium Beauty Serum",
            seller=SellerInfo(name="trusted beauty store", rating=4.8, review_count=900),
            reviews=[],
        )
    )
    weak = service.score_product(
        ProductPageData(
            url="https://example.com/p2",
            product_title="Generic Item",
            seller=SellerInfo(name="unknown shop", rating=2.4, review_count=2),
            reviews=[],
        )
    )

    assert reliable.seller_mode == "artifact_v2"
    assert weak.seller_mode == "artifact_v2"
    assert reliable.seller_reliability > weak.seller_reliability
    assert service.artifact_status["seller"] == "loaded"


def test_v1_text_only_model_still_supported(tmp_path: Path) -> None:
    model_path = _make_v1_seller_model(tmp_path)
    service = RiskModelService(seller_model_path=str(model_path), price_model_path=None, policy_model_path=None)
    result = service.score_product(
        ProductPageData(
            url="https://example.com/p3",
            product_title="trusted beauty store premium",
            seller=SellerInfo(name="trusted beauty store", rating=4.7, review_count=500),
            reviews=[],
        )
    )
    assert result.seller_mode == "artifact"
    assert 0 <= result.seller_reliability <= 100


def test_missing_artifact_falls_back_to_rules() -> None:
    service = RiskModelService(seller_model_path=None, price_model_path=None, policy_model_path=None)
    result = service.score_product(
        ProductPageData(url="https://example.com/p4", product_title="Anything", reviews=[])
    )
    assert result.seller_mode == "rule_fallback"
