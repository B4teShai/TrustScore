"""TrustScore formula, risk classification, and explanation generation."""

from __future__ import annotations

from app.core.config import settings
from app.schemas.product_analysis import ComponentScores


REASON_TEXT: dict[str, tuple[str, str, str]] = {
    "review_authenticity": (
        "Review patterns look mostly natural.",
        "Some review patterns need closer checking.",
        "Some reviews look repeated or suspicious.",
    ),
    "seller_reliability": (
        "Seller information looks strong.",
        "Seller reliability has mixed visible signals.",
        "Seller reliability is weak.",
    ),
    "sentiment": (
        "Review sentiment is mostly positive.",
        "Review sentiment is mixed.",
        "Several reviews mention product quality problems.",
    ),
    "return_policy_clarity": (
        "Return policy wording looks clear.",
        "Return policy may need manual checking.",
        "Return policy is unclear or not visible.",
    ),
    "price_safety": (
        "Price does not show an obvious market-risk signal.",
        "Price should be compared with similar products.",
        "Product price looks unusual compared with market price.",
    ),
    "user_feedback_history": (
        "Prior feedback does not add extra risk.",
        "User feedback history is still limited.",
        "User feedback history is too limited to reduce risk.",
    ),
}


def clamp_score(value: float) -> int:
    """Keep public scores inside the 0 to 100 range."""
    return max(0, min(100, round(value)))


def calculate_trustscore(scores: ComponentScores, *, include_feedback: bool = False) -> int:
    """Apply active TrustScore weights and normalize to a 0-100 public score.

    Feedback is excluded by default because the runtime currently stores
    feedback for evaluation but does not produce a reviewed feedback-history
    signal. This avoids giving a hardcoded neutral value real score weight.
    """
    weights = {
        "review_authenticity": settings.trust_weight_review_authenticity,
        "seller_reliability": settings.trust_weight_seller_reliability,
        "sentiment": settings.trust_weight_sentiment,
        "return_policy_clarity": settings.trust_weight_policy,
        "price_safety": settings.trust_weight_price,
    }
    if include_feedback and settings.trust_weight_feedback > 0:
        weights["user_feedback_history"] = settings.trust_weight_feedback

    active = {key: weight for key, weight in weights.items() if weight > 0}
    total_weight = sum(active.values())
    if total_weight <= 0:
        return 50

    score_values = scores.model_dump()
    weighted_score = sum(score_values[key] * weight for key, weight in active.items())
    return clamp_score(weighted_score / total_weight)


def classify_risk(score: int) -> str:
    """Convert a TrustScore into the public risk label."""
    if score >= 80:
        return "Low Risk"
    if score >= 50:
        return "Medium Risk"
    return "High Risk"


def top_reasons(scores: ComponentScores) -> list[str]:
    """Explain the three weakest component scores."""
    score_values = {
        key: value
        for key, value in scores.model_dump().items()
        if key != "user_feedback_history"
    }
    weakest_components = sorted(score_values.items(), key=lambda item: item[1])[:3]
    return [_reason_for_score(component, score) for component, score in weakest_components]


def recommendation_for_risk(risk_level: str) -> str:
    """Return non-accusatory shopping guidance for the final risk band."""
    if risk_level == "Low Risk":
        return "Signals look healthy. Still review seller and return details before buying."
    if risk_level == "Medium Risk":
        return "Check seller details, recent reviews, and return policy before buying."
    return "Use caution. Verify the seller, reviews, and return policy before trusting this listing."


def _reason_for_score(component: str, score: int) -> str:
    positive, mixed, negative = REASON_TEXT[component]
    if score >= 80:
        return positive
    if score >= 50:
        return mixed
    return negative
