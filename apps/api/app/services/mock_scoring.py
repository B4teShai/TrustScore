"""Temporary mock scoring for the backend MVP.

This file intentionally avoids real ML. The values are randomized so the
extension can exercise loading, refresh, and result-display states against a
real API response while later ML services are still unfinished.
"""

from random import choice, randint, uniform
from uuid import uuid4

from app.core.config import settings
from app.schemas.product_analysis import (
    ComponentScores,
    ProductMetadata,
    ProductPageData,
    ProductAnalysisResponse,
)


def _clamp_score(value: float) -> int:
    """Keep every score inside the user-facing 0 to 100 range."""
    return max(0, min(100, round(value)))


def _risk_level(score: int) -> str:
    """Convert a TrustScore into the MVP risk label."""
    if score >= 80:
        return "Low Risk"
    if score >= 50:
        return "Medium Risk"
    return "High Risk"


def _data_completeness(payload: ProductPageData) -> float:
    """Estimate how much useful product data the frontend sent."""
    available_parts = [
        payload.price is not None,
        payload.seller is not None,
        bool(payload.return_policy),
        bool(payload.reviews),
        payload.rating is not None or payload.review_count is not None,
    ]
    return sum(available_parts) / len(available_parts)


def _random_component_scores(risk_target: str) -> ComponentScores:
    """Generate component scores that look consistent with a risk scenario."""
    ranges = {
        "Low Risk": {
            "review_authenticity": (82, 96),
            "seller_reliability": (82, 98),
            "sentiment": (78, 94),
            "return_policy_clarity": (80, 95),
            "price_safety": (78, 95),
            "user_feedback_history": (70, 90),
        },
        "Medium Risk": {
            "review_authenticity": (52, 78),
            "seller_reliability": (55, 82),
            "sentiment": (50, 78),
            "return_policy_clarity": (45, 76),
            "price_safety": (55, 82),
            "user_feedback_history": (50, 72),
        },
        "High Risk": {
            "review_authenticity": (18, 48),
            "seller_reliability": (20, 50),
            "sentiment": (24, 52),
            "return_policy_clarity": (15, 45),
            "price_safety": (22, 52),
            "user_feedback_history": (35, 55),
        },
    }
    selected = ranges[risk_target]
    return ComponentScores(
        review_authenticity=randint(*selected["review_authenticity"]),
        seller_reliability=randint(*selected["seller_reliability"]),
        sentiment=randint(*selected["sentiment"]),
        return_policy_clarity=randint(*selected["return_policy_clarity"]),
        price_safety=randint(*selected["price_safety"]),
        user_feedback_history=randint(*selected["user_feedback_history"]),
    )


def _weighted_trust_score(scores: ComponentScores) -> int:
    """Apply the documented TrustScore weights to component scores."""
    return _clamp_score(
        scores.review_authenticity * settings.trust_weight_review_authenticity
        + scores.seller_reliability * settings.trust_weight_seller_reliability
        + scores.sentiment * settings.trust_weight_sentiment
        + scores.return_policy_clarity * settings.trust_weight_policy
        + scores.price_safety * settings.trust_weight_price
        + scores.user_feedback_history * settings.trust_weight_feedback
    )


def _top_reasons(scores: ComponentScores, risk_level: str) -> list[str]:
    reason_map = {
        "review_authenticity": {
            "Low Risk": "Review patterns look mostly natural in this mock scan.",
            "Medium Risk": "Some review patterns need closer checking.",
            "High Risk": "Several review patterns look suspicious in this mock scan.",
        },
        "seller_reliability": {
            "Low Risk": "Seller information looks strong for this mock scan.",
            "Medium Risk": "Seller reliability has mixed visible signals.",
            "High Risk": "Seller reliability appears weak in this mock scan.",
        },
        "sentiment": {
            "Low Risk": "Review sentiment is mostly positive.",
            "Medium Risk": "Review sentiment is mixed.",
            "High Risk": "Reviews include several negative quality signals.",
        },
        "return_policy_clarity": {
            "Low Risk": "Return policy wording looks clear.",
            "Medium Risk": "Return policy may need manual checking.",
            "High Risk": "Return policy looks unclear or incomplete.",
        },
        "price_safety": {
            "Low Risk": "Price does not show obvious risk in this mock scan.",
            "Medium Risk": "Price should be compared with similar products.",
            "High Risk": "Price looks unusually risky in this mock scan.",
        },
        "user_feedback_history": {
            "Low Risk": "Previous feedback signal is healthy in this mock scan.",
            "Medium Risk": "User feedback history is still limited.",
            "High Risk": "User feedback history does not reduce the risk.",
        },
    }
    score_values = scores.model_dump()
    lowest_components = sorted(score_values, key=score_values.get)[:3]
    return [reason_map[name][risk_level] for name in lowest_components]


def _recommendation(risk_level: str) -> str:
    recommendations = {
        "Low Risk": [
            "This mock result looks low risk. Still review seller and return details before buying.",
            "Signals look healthy in this demo response. Continue with normal shopping checks.",
        ],
        "Medium Risk": [
            "Check seller details, recent reviews, and return policy before buying.",
            "Compare this product with alternatives before making a purchase decision.",
        ],
        "High Risk": [
            "Consider avoiding this product unless you can verify the seller and return policy.",
            "Use caution. Review suspicious signals before trusting this listing.",
        ],
    }
    return choice(recommendations[risk_level])


def build_mock_trustscore(
    payload: ProductPageData,
) -> ProductAnalysisResponse:
    """Build one randomized mock TrustScore response for UI/API testing."""
    risk_target = choice(["Low Risk", "Medium Risk", "High Risk"])
    scores = _random_component_scores(risk_target)
    trust_score = _weighted_trust_score(scores)
    risk_level = _risk_level(trust_score)

    # More complete payloads should feel slightly more reliable even in mock mode.
    completeness = _data_completeness(payload)
    confidence = round(min(0.96, uniform(0.48, 0.78) + completeness * 0.18), 2)

    return ProductAnalysisResponse(
        scan_id=str(uuid4()),
        product=ProductMetadata(
            url=payload.url,
            site=payload.site,
            product_title=payload.product_title,
            product_image_url=payload.product_image_url,
            price=payload.price,
            currency=payload.currency,
            seller_name=payload.seller.name if payload.seller else None,
        ),
        trust_score=trust_score,
        risk_level=risk_level,
        confidence=confidence,
        component_scores=scores,
        top_reasons=_top_reasons(scores, risk_level),
        recommendation=_recommendation(risk_level),
        model_version=settings.model_version,
        is_mock=True,
    )
