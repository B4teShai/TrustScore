"""Product analysis API routes."""

import ipaddress
from urllib.parse import urlparse, urlunparse

from fastapi import APIRouter, HTTPException

from app.core.config import settings
from app.db.repository import save_feedback, save_scan
from app.ml.fake_review_service import fake_review_service
from app.ml.risk_service import risk_model_service
from app.ml.sentiment_service import sentiment_service
from app.page_fetching.fetcher import PageFetchError, URLValidationError, validate_public_web_url
from app.schemas.product_analysis import (
    ExtractedProductData,
    ExtractedProductScanRequest,
    FeedbackRequest,
    FeedbackResponse,
    ProductAnalysisRequest,
    ProductAnalysisResponse,
    ProductPageData,
)
from app.services.product_page_analysis import ProductNotDetectedError, analyze_product_url
from app.services.product_scoring import build_trustscore


router = APIRouter(prefix="/api", tags=["product analysis"])


@router.post("/analyze-product", response_model=ProductAnalysisResponse)
def analyze_product(payload: ProductAnalysisRequest) -> ProductAnalysisResponse:
    """Deprecated compatibility alias for URL-only product analysis."""
    return _analyze_product_url(payload)


@router.post("/v1/scan", response_model=ProductAnalysisResponse)
def scan_product(payload: ProductAnalysisRequest) -> ProductAnalysisResponse:
    """Documented scan route for product-page TrustScore analysis."""
    return _analyze_product_url(payload)


@router.post("/v1/scan-extracted", response_model=ProductAnalysisResponse)
def scan_extracted_product(payload: ExtractedProductScanRequest) -> ProductAnalysisResponse:
    """Score active-tab product fields when server-side fetching is blocked."""
    try:
        product = _sanitize_extracted_product(payload.product)
    except URLValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_product_url", "message": str(exc)},
        ) from exc

    response = build_trustscore(
        product,
        fetch_mode="extension_dom",
        extraction_signals=_signals_from_extracted_product(product),
        locale=payload.locale,
    )
    save_scan(
        product=product,
        response=response,
        browser_id=payload.browser_id,
        fetch_mode=response.fetch_mode,
        extraction_signals=response.extraction_signals,
    )
    return response


@router.post("/v1/feedback", response_model=FeedbackResponse)
def submit_feedback(payload: FeedbackRequest) -> FeedbackResponse:
    """Store feedback when persistence is configured; otherwise accept local mode."""
    result = save_feedback(payload)
    if result.status == "missing":
        raise HTTPException(
            status_code=404,
            detail={"code": "scan_not_found", "message": "Scan ID was not found."},
        )
    if result.status == "error":
        raise HTTPException(
            status_code=503,
            detail={
                "code": "feedback_persistence_unavailable",
                "message": "Feedback persistence is temporarily unavailable.",
            },
        )
    return FeedbackResponse(status=result.status)


@router.get("/v1/model-info")
def model_info() -> dict[str, object]:
    """Return runtime model and scoring configuration."""
    return {
        "model_version": settings.model_version,
        "model_version_tag": settings.model_version_tag,
        "score_semantics": (
            "TrustScore is normalized over active non-feedback weights. "
            "Feedback is collected for evaluation and not used until reviewed aggregates exist."
        ),
        "sentiment_model": settings.sentiment_model_name,
        "sentiment_mode": "artifact_or_transformers_if_cached_else_fallback",
        "sentiment_artifact_status": sentiment_service.artifact_status,
        "fake_review_model": "v3_calibrated_tfidf_if_artifacts_exist_else_heuristic",
        "fake_review_artifact_status": fake_review_service.artifact_status,
        "risk_model_artifact_status": risk_model_service.artifact_status,
        "trustscore_weights": {
            "review_authenticity": settings.trust_weight_review_authenticity,
            "seller_reliability": settings.trust_weight_seller_reliability,
            "sentiment": settings.trust_weight_sentiment,
            "return_policy_clarity": settings.trust_weight_policy,
            "price_safety": settings.trust_weight_price,
            "user_feedback_history": settings.trust_weight_feedback,
        },
        "feedback_scoring": "not_applied",
        "ai_feedback": {
            "active": settings.ai_feedback_active,
            "model": settings.anthropic_model if settings.ai_feedback_active else None,
            "source_when_inactive": "rule",
        },
        "scan_persistence": "database" if settings.database_url else (
            "local_jsonl" if settings.persist_local_scans else "disabled"
        ),
    }


def _analyze_product_url(payload: ProductAnalysisRequest) -> ProductAnalysisResponse:
    try:
        analysis = analyze_product_url(payload.url)
    except URLValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_product_url", "message": str(exc)},
        ) from exc
    except ProductNotDetectedError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "product_not_detected", "message": str(exc)},
        ) from exc
    except PageFetchError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "product_page_unavailable", "message": str(exc)},
        ) from exc

    response = build_trustscore(
        analysis.product,
        fetch_mode=analysis.fetch_mode,
        extraction_signals=analysis.signals,
        locale=payload.locale,
    )
    save_scan(
        product=analysis.product,
        response=response,
        browser_id=payload.browser_id,
        fetch_mode=analysis.fetch_mode,
        extraction_signals=analysis.signals,
    )
    return response


def _sanitize_extracted_product(product: ExtractedProductData) -> ProductPageData:
    safe_url = _validate_public_reference_url(product.url)
    safe_image_url = None
    if product.product_image_url:
        try:
            safe_image_url = _validate_public_reference_url(product.product_image_url)
        except ValueError:
            safe_image_url = None
    seller = None
    if product.seller is not None:
        seller = {
            "name": _truncate(product.seller.name, 160),
            "rating": product.seller.rating,
            "review_count": product.seller.review_count,
            "years_active": product.seller.years_active,
        }
    return ProductPageData(
        url=safe_url,
        site=_truncate(product.site, 255),
        product_title=_truncate(product.product_title, 240) or "Unknown product",
        description=_truncate(product.description, 1000),
        product_image_url=safe_image_url,
        price=product.price,
        currency=_truncate(product.currency, 16),
        average_market_price=product.average_market_price,
        seller=seller,
        return_policy=_truncate(product.return_policy, 1000),
        reviews=product.reviews[:50],
        rating=product.rating,
        review_count=product.review_count,
        units_bought_recent=product.units_bought_recent,
    )


def _validate_public_reference_url(raw_url: str) -> str:
    """Validate URLs supplied by active-tab extraction without resolving DNS."""
    try:
        parsed = urlparse(raw_url.strip())
        port = parsed.port
    except ValueError as exc:
        raise URLValidationError("URL port is invalid.") from exc

    if parsed.scheme.lower() not in {"http", "https"}:
        raise URLValidationError("Only HTTP and HTTPS product-page URLs are supported.")
    if not parsed.hostname:
        raise URLValidationError("URL must include a hostname.")
    if parsed.username or parsed.password:
        raise URLValidationError("URLs with embedded credentials are not supported.")
    if port is not None and port not in {80, 443}:
        raise URLValidationError("Only default HTTP and HTTPS ports are supported.")

    hostname = parsed.hostname.strip().lower().rstrip(".")
    if (
        hostname in {"localhost", "localhost.localdomain"}
        or hostname.endswith(".localhost")
        or hostname.endswith(".local")
    ):
        raise URLValidationError("Local machine URLs are not supported.")

    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        address = None
    if address is not None and (
        address.is_loopback
        or address.is_private
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    ):
        raise URLValidationError("Only public web URLs are supported.")

    normalized = parsed._replace(
        scheme=parsed.scheme.lower(),
        netloc=hostname if port is None else f"{hostname}:{port}",
        fragment="",
    )
    return urlunparse(normalized)


def _truncate(value: str | None, max_length: int) -> str | None:
    if value is None:
        return None
    compact = " ".join(value.split())
    if not compact:
        return None
    return compact[:max_length]


def _signals_from_extracted_product(product: ProductPageData) -> list[str]:
    signals = ["extension_dom", "title"]
    if product.price is not None:
        signals.append("price")
    if product.product_image_url:
        signals.append("image")
    if product.seller and product.seller.name:
        signals.append("seller")
    if product.rating is not None or product.review_count is not None:
        signals.append("rating_or_review_count")
    if product.reviews:
        signals.append("review_text")
    if product.return_policy:
        signals.append("policy")
    return signals
