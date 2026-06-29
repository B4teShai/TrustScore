import json
from pathlib import Path

from app.ml.fake_review_service import FakeReviewService
from app.ml.preprocessing import clean_review_text, extract_review_features
from app.ml.risk_service import RiskModelService
from app.schemas.product_analysis import ProductPageData, ReviewInput
from app.services.trustscore_engine import calculate_trustscore, classify_risk, top_reasons
from app.schemas.product_analysis import ComponentScores


def test_clean_review_text_removes_html_and_normalizes_whitespace() -> None:
    assert clean_review_text(" <b>Great</b>\n product&nbsp; ") == "great product"


def test_review_features_detect_duplicates_and_suspicious_patterns() -> None:
    features = extract_review_features(
        [
            ReviewInput(text="Great", rating=5, verified_purchase=False),
            ReviewInput(text="Great", rating=5, verified_purchase=False),
            ReviewInput(text="Poor quality and broken", rating=5, verified_purchase=True),
            ReviewInput(text="   ", rating=1),
        ]
    )

    assert features.total_reviews == 3
    assert features.unique_review_count == 2
    assert features.duplicate_review_rate > 0
    assert features.short_five_star_rate > 0
    assert features.negative_keyword_rate > 0
    assert features.rating_sentiment_mismatch_rate > 0


def test_fake_review_fallback_penalizes_repeated_short_five_star_reviews() -> None:
    service = FakeReviewService(model_path=None, vectorizer_path=None)
    suspicious = extract_review_features(
        [
            ReviewInput(text="Great", rating=5, verified_purchase=False),
            ReviewInput(text="Great", rating=5, verified_purchase=False),
            ReviewInput(text="Great", rating=5, verified_purchase=False),
        ]
    )
    varied = extract_review_features(
        [
            ReviewInput(
                text="Good build quality and fast delivery",
                rating=5,
                verified_purchase=True,
            ),
            ReviewInput(
                text="Works as expected after a week",
                rating=4,
                verified_purchase=True,
            ),
        ]
    )

    assert service.score_product(suspicious).authenticity_score < service.score_product(
        varied
    ).authenticity_score


def test_fake_review_artifact_failure_falls_back_deterministically() -> None:
    class BrokenArtifactService(FakeReviewService):
        @property
        def _artifacts(self):
            return object(), object()

        def _score_with_artifacts(self, features):
            raise RuntimeError("artifact failed")

    service = BrokenArtifactService(model_path="model.joblib", vectorizer_path="vectorizer.joblib")
    features = extract_review_features(
        [
            ReviewInput(text="Great", rating=5, verified_purchase=False),
            ReviewInput(text="Poor quality and broken", rating=5, verified_purchase=True),
        ]
    )

    result = service.score_product(features)

    assert result.mode == "heuristic_fallback"
    assert 0 <= result.authenticity_score <= 100


def test_rule_based_risk_scores_handle_missing_and_market_price() -> None:
    service = RiskModelService(
        seller_model_path=None,
        price_model_path=None,
        policy_model_path=None,
    )
    missing = service.score_product(
        ProductPageData(url="https://example.com/product/1", product_title="Unknown Product", reviews=[])
    )
    priced = service.score_product(
        ProductPageData(
            url="https://example.com/product/2",
            product_title="Cheap Product",
            price=10,
            average_market_price=50,
            return_policy="30-day return and refund warranty",
            reviews=[],
        )
    )

    assert missing.seller_reliability == 50
    assert missing.price_safety == 50
    assert missing.return_policy_clarity == 50
    assert missing.seller_mode == "rule_fallback"
    assert missing.price_mode == "rule_fallback"
    assert missing.policy_mode == "rule_fallback"
    assert priced.price_safety == 35
    assert priced.return_policy_clarity >= 80


def test_marketplace_popularity_lifts_seller_reliability_without_seller_fields() -> None:
    service = RiskModelService(
        seller_model_path=None,
        price_model_path=None,
        policy_model_path=None,
    )
    unknown = service.score_product(
        ProductPageData(
            url="https://example.com/product/x",
            product_title="Obscure Listing",
            seller={"name": "Tiny Store"},
            reviews=[],
        )
    )
    popular = service.score_product(
        ProductPageData(
            url="https://example.com/product/y",
            product_title="Best Seller",
            seller={"name": "VIFUUR Store"},
            rating=4.3,
            review_count=147552,
            units_bought_recent=8000,
            reviews=[],
        )
    )

    # A name-only seller stays near neutral; high rating volume + recent sales
    # lift seller reliability and its completeness without inventing seller data.
    assert unknown.seller_reliability <= 60
    assert popular.seller_reliability >= 75
    assert popular.seller_completeness > unknown.seller_completeness


def test_trustscore_formula_thresholds_and_reasons() -> None:
    scores = ComponentScores(
        review_authenticity=100,
        seller_reliability=50,
        sentiment=50,
        return_policy_clarity=50,
        price_safety=50,
        user_feedback_history=50,
    )

    assert calculate_trustscore(scores) == 66
    assert classify_risk(85) == "Low Risk"
    assert classify_risk(50) == "Medium Risk"
    assert classify_risk(49) == "High Risk"
    assert len(top_reasons(scores)) == 3


def test_saved_model_metadata_meets_current_smoke_gates() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    fake_metadata = json.loads(
        (repo_root / "ml" / "artifacts" / "fake_full" / "model_metadata.json").read_text()
    )
    sentiment_metadata = json.loads(
        (
            repo_root
            / "ml"
            / "artifacts"
            / "sentiment"
            / "sentiment_model_metadata.json"
        ).read_text()
    )

    assert fake_metadata["metrics"]["recall"] >= 0.80
    assert fake_metadata["metrics"]["f1"] >= 0.80
    assert sentiment_metadata["metrics"]["f1_macro"] >= 0.70
