"""Pydantic models for product analysis requests and responses."""

from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


RiskLevel = Literal["Low Risk", "Medium Risk", "High Risk"]
FeedbackStatus = Literal["saved", "accepted"]
TargetMarket = Literal["auto", "US", "JP", "EU", "UK"]
ComponentKey = Literal[
    "review_authenticity",
    "seller_reliability",
    "sentiment",
    "return_policy_clarity",
    "price_safety",
    "user_feedback_history",
]
FeedbackIssueCategory = Literal[
    "score_too_high",
    "score_too_low",
    "wrong_product",
    "wrong_seller",
    "wrong_reviews",
    "wrong_price",
    "wrong_policy",
    "missing_evidence",
    "other",
]


class StrictBaseModel(BaseModel):
    """Base schema that rejects unexpected request or response fields."""

    model_config = ConfigDict(extra="forbid")


class LenientBaseModel(BaseModel):
    """Boundary schema for browser-extracted fields that are sanitized later."""

    model_config = ConfigDict(extra="ignore")


class SellerInfo(StrictBaseModel):
    """Seller details collected from a product page when visible."""

    name: str | None = Field(default=None, max_length=160, examples=["Example Store"])
    rating: float | None = Field(default=None, ge=0, le=5, examples=[4.6])
    review_count: int | None = Field(default=None, ge=0, examples=[5000])
    years_active: int | None = Field(default=None, ge=0, examples=[3])


class ReviewInput(StrictBaseModel):
    """Single visible review sample from the product page."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        examples=["Good quality and fast delivery."],
    )
    rating: float | None = Field(default=None, ge=0, le=5, examples=[5])
    date: str | None = Field(default=None, max_length=64, examples=["2026-01-01"])
    verified_purchase: bool | None = Field(default=None, examples=[True])


class ProductAnalysisRequest(StrictBaseModel):
    """URL-only payload accepted by the product-analysis endpoint."""

    url: str = Field(
        ...,
        min_length=1,
        max_length=8192,
        examples=["https://example.com/product/123"],
    )
    browser_id: str | None = Field(
        default=None,
        max_length=128,
        examples=["anonymous-browser-id"],
    )
    locale: str | None = Field(
        default=None,
        max_length=35,
        description="Page language (BCP-47), used to localize reasons and guidance.",
        examples=["ja-JP", "en-US"],
    )
    target_market: TargetMarket = Field(
        default="auto",
        description="Target shopping market used to normalize marketplace currency.",
        examples=["US", "JP", "EU", "UK"],
    )


class ProductPageData(StrictBaseModel):
    """Product data extracted by the backend from a public product page."""

    url: str = Field(..., max_length=8192, examples=["https://example.com/product/123"])
    site: str | None = Field(default=None, max_length=255, examples=["example-shop"])
    product_title: str = Field(
        ...,
        min_length=1,
        max_length=240,
        examples=["Wireless Headphones"],
    )
    description: str | None = Field(
        default=None,
        max_length=1000,
        examples=["Bluetooth noise cancelling headphones"],
    )
    product_image_url: str | None = Field(
        default=None,
        max_length=4096,
        examples=["https://example.com/product-image.jpg"],
    )
    price: float | None = Field(default=None, ge=0, examples=[29.99])
    currency: str | None = Field(default=None, max_length=16, examples=["USD"])
    average_market_price: float | None = Field(default=None, gt=0, examples=[59.99])
    market_reference_count: int | None = Field(default=None, ge=0, examples=[12])
    market_reference_source: str | None = Field(default=None, max_length=80, examples=["Serper"])
    market_reference_original_currency: str | None = Field(
        default=None,
        max_length=16,
        examples=["USD"],
    )
    market_reference_exchange_rate: float | None = Field(default=None, gt=0, examples=[156.2])
    market_reference_exchange_rate_source: str | None = Field(
        default=None,
        max_length=80,
        examples=["Frankfurter"],
    )
    market_reference_exchange_rate_date: str | None = Field(
        default=None,
        max_length=32,
        examples=["2026-06-30"],
    )
    seller: SellerInfo | None = None
    return_policy: str | None = Field(
        default=None,
        max_length=1000,
        examples=["30-day return policy available"],
    )
    reviews: list[ReviewInput] = Field(default_factory=list, max_length=50)
    rating: float | None = Field(default=None, ge=0, le=5, examples=[4.3])
    review_count: int | None = Field(default=None, ge=0, examples=[120])
    units_bought_recent: int | None = Field(
        default=None,
        ge=0,
        description="Recent purchase volume from a marketplace badge (e.g. '8K+ bought in past month').",
        examples=[8000],
    )


class ExtractedSellerInfo(LenientBaseModel):
    """Lenient seller details accepted from an active-tab preview."""

    name: str | None = Field(default=None, max_length=1000)
    rating: float | None = Field(default=None, ge=0)
    review_count: int | None = Field(default=None, ge=0)
    years_active: int | None = Field(default=None, ge=0)


class ExtractedReviewInput(LenientBaseModel):
    """Lenient review details accepted from an active-tab preview."""

    text: str = Field(..., min_length=1, max_length=4000)
    rating: float | None = Field(default=None, ge=0)
    date: str | None = Field(default=None, max_length=128)
    verified_purchase: bool | None = None


class ExtractedProductData(LenientBaseModel):
    """Lenient product fields accepted from the active-tab preview boundary."""

    url: str = Field(..., min_length=1, max_length=8192)
    site: str | None = Field(default=None, max_length=512)
    product_title: str = Field(..., min_length=1, max_length=1000)
    description: str | None = Field(default=None, max_length=4000)
    product_image_url: str | None = Field(default=None, max_length=4096)
    price: float | None = Field(default=None, ge=0)
    currency: str | None = Field(default=None, max_length=64)
    average_market_price: float | None = Field(default=None, gt=0)
    seller: ExtractedSellerInfo | None = None
    return_policy: str | None = Field(default=None, max_length=4000)
    reviews: list[ExtractedReviewInput] = Field(default_factory=list, max_length=50)
    rating: float | None = Field(default=None, ge=0)
    review_count: int | None = Field(default=None, ge=0)
    units_bought_recent: int | None = Field(default=None, ge=0)


class ExtractedProductScanRequest(LenientBaseModel):
    """Minimal active-tab product data used when server-side fetching is blocked."""

    product: ExtractedProductData
    browser_id: str | None = Field(
        default=None,
        max_length=128,
        examples=["anonymous-browser-id"],
    )
    locale: str | None = Field(
        default=None,
        max_length=35,
        description="Page language (BCP-47), used to localize reasons and guidance.",
        examples=["ja-JP", "en-US"],
    )
    target_market: TargetMarket = Field(
        default="auto",
        description="Target shopping market used to normalize marketplace currency.",
        examples=["US", "JP", "EU", "UK"],
    )


class ProductMetadata(StrictBaseModel):
    """Public product summary returned with a TrustScore result."""

    url: str = Field(..., max_length=8192)
    site: str | None = Field(default=None, max_length=255)
    product_title: str = Field(..., min_length=1, max_length=240)
    product_image_url: str | None = Field(default=None, max_length=4096)
    price: float | None = None
    currency: str | None = Field(default=None, max_length=16)
    seller_name: str | None = Field(default=None, max_length=160)


class ComponentScores(StrictBaseModel):
    """Breakdown of the factors used in the TrustScore result."""

    review_authenticity: int = Field(..., ge=0, le=100)
    seller_reliability: int = Field(..., ge=0, le=100)
    sentiment: int = Field(..., ge=0, le=100)
    return_policy_clarity: int = Field(..., ge=0, le=100)
    price_safety: int = Field(..., ge=0, le=100)
    user_feedback_history: int = Field(..., ge=0, le=100)


class ComponentEvidence(StrictBaseModel):
    """Evidence and missing inputs behind one public component score."""

    component: ComponentKey
    summary: str = Field(..., max_length=240)
    evidence: list[str] = Field(default_factory=list, max_length=5)
    missing_inputs: list[str] = Field(default_factory=list, max_length=8)
    confidence: float = Field(..., ge=0, le=1)


class ProductAnalysisResponse(StrictBaseModel):
    """Structured analysis response returned to the extension frontend."""

    scan_id: UUID
    product: ProductMetadata
    trust_score: int = Field(..., ge=0, le=100)
    risk_level: RiskLevel
    confidence: float = Field(..., ge=0, le=1)
    component_scores: ComponentScores
    top_reasons: list[str] = Field(default_factory=list, max_length=3)
    evidence: list[ComponentEvidence] = Field(default_factory=list, max_length=6)
    missing_inputs: list[str] = Field(default_factory=list, max_length=20)
    score_semantics: str = Field(
        default=(
            "TrustScore is a weighted signal score from available evidence. "
            "Confidence and missing inputs should be considered before relying on it."
        ),
        max_length=500,
    )
    recommendation: str = Field(..., max_length=500)
    recommendation_source: Literal["rule", "ai"] = "rule"
    language: str = Field(default="en", max_length=16)
    model_version: str
    fetch_mode: str = "unknown"
    extraction_signals: list[str] = Field(default_factory=list, max_length=20)
    model_modes: dict[str, str] = Field(default_factory=dict)
    model_artifact_status: dict[str, Any] = Field(default_factory=dict)
    model_versions: dict[str, str] = Field(default_factory=dict)
    is_mock: bool = False


class FeedbackRequest(StrictBaseModel):
    """User feedback for a completed scan."""

    scan_id: UUID
    browser_id: str | None = Field(default=None, max_length=128)
    helpful: bool
    issue_category: FeedbackIssueCategory | None = None
    corrected_component: ComponentKey | None = None
    expected_risk_level: RiskLevel | None = None
    comment: str | None = Field(default=None, max_length=1000)


class FeedbackResponse(StrictBaseModel):
    """Feedback acknowledgement with explicit persistence semantics."""

    status: FeedbackStatus = "saved"
