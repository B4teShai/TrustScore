"""Product analysis API routes."""

import ipaddress
from urllib.parse import urlparse, urlunparse

from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.core.config import settings
from app.db.repository import save_feedback, save_scan
from app.extraction.product_page import clean_policy_snippet, clean_review_body
from app.ml.fake_review_service import fake_review_service
from app.ml.risk_service import risk_model_service
from app.ml.sentiment_service import sentiment_service
from app.page_fetching.fetcher import PageFetchError, URLValidationError, validate_public_web_url
from app.schemas.product_analysis import (
    ExtractedReviewInput,
    ExtractedProductData,
    ExtractedProductScanRequest,
    FeedbackRequest,
    FeedbackResponse,
    ProductAnalysisRequest,
    ProductAnalysisResponse,
    ProductPageData,
    ReviewInput,
)
from app.services.market_context import (
    expected_currency_for_market,
    normalize_currency_code,
    resolve_target_market,
    should_use_price_currency,
)
from app.services.market_reference import enrich_market_reference
from app.services.product_page_analysis import ProductNotDetectedError, analyze_product_url
from app.services.product_scoring import build_trustscore


router = APIRouter(prefix="/api", tags=["product analysis"])


def _require_amazon_host(raw_url: str) -> None:
    """Reject non-Amazon URLs; TrustScore supports Amazon product pages only."""
    try:
        host = (urlparse(raw_url).hostname or "").lower()
    except ValueError:
        host = ""
    if "amazon." not in host:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "unsupported_site",
                "message": "TrustScore currently supports Amazon product pages only.",
            },
        )


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
    _require_amazon_host(payload.product.url)
    try:
        product, extra_signals = _sanitize_extracted_product(
            payload.product,
            target_market=payload.target_market,
            locale=payload.locale,
        )
    except URLValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_product_url", "message": str(exc)},
        ) from exc

    product, market_signals = enrich_market_reference(
        product,
        target_market=payload.target_market,
        locale=payload.locale,
    )
    extraction_signals = [
        *_signals_from_extracted_product(product),
        *extra_signals,
        *market_signals,
    ]
    page_type = _page_type_from_url(product.url)
    response = build_trustscore(
        product,
        fetch_mode="extension_dom",
        extraction_signals=extraction_signals,
        locale=payload.locale,
        target_market=payload.target_market,
        page_type=page_type,
        product_identity_confidence=_identity_confidence_from_signals(extraction_signals, page_type),
        canonical_product_url=None,
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
def submit_feedback(
    payload: FeedbackRequest,
    background_tasks: BackgroundTasks,
) -> FeedbackResponse:
    """Store feedback when persistence is configured; otherwise accept local mode."""
    if settings.database_url:
        background_tasks.add_task(save_feedback, payload)
        return FeedbackResponse(status="accepted")

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
        "feedback_scoring": (
            "applied_small_weight_when_feedback_present"
            if settings.trust_weight_feedback > 0
            else "not_applied"
        ),
        "ai_feedback": {
            "active": settings.ai_feedback_active,
            "model": settings.anthropic_model if settings.ai_feedback_active else None,
            "source_when_inactive": "rule",
        },
        "market_reference": {
            "provider": "serper",
            "active": bool(settings.market_reference_enabled and settings.serper_api_key),
            "cache_ttl_seconds": settings.market_reference_cache_ttl_seconds,
            "min_results": settings.market_reference_min_results,
            "fx_conversion": {
                "provider": "frankfurter",
                "active": settings.exchange_rate_enabled,
            },
        },
        "scan_persistence": "database" if settings.database_url else (
            "local_jsonl" if settings.persist_local_scans else "disabled"
        ),
    }


def _analyze_product_url(payload: ProductAnalysisRequest) -> ProductAnalysisResponse:
    _require_amazon_host(payload.url)
    try:
        analysis = analyze_product_url(
            payload.url,
            target_market=payload.target_market,
            locale=payload.locale,
        )
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
        target_market=payload.target_market,
        page_type=analysis.page_type,
        product_identity_confidence=analysis.product_identity_confidence,
        canonical_product_url=analysis.canonical_product_url,
    )
    save_scan(
        product=analysis.product,
        response=response,
        browser_id=payload.browser_id,
        fetch_mode=analysis.fetch_mode,
        extraction_signals=analysis.signals,
    )
    return response


def _sanitize_extracted_product(
    product: ExtractedProductData,
    *,
    target_market: str | None = "auto",
    locale: str | None = None,
) -> tuple[ProductPageData, list[str]]:
    safe_url = _validate_public_reference_url(product.url)
    resolved_market = resolve_target_market(target_market, url=safe_url, locale=locale)
    expected_currency = expected_currency_for_market(resolved_market)
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
            "rating": _rating_or_none(product.seller.rating),
            "review_count": product.seller.review_count,
            "years_active": product.seller.years_active,
            "sold_by": _truncate(product.seller.sold_by, 160),
            "ships_from": _truncate(product.seller.ships_from, 160),
            "fulfilled_by": _truncate(product.seller.fulfilled_by, 160),
            "brand_store_name": _truncate(product.seller.brand_store_name, 160),
            "is_platform_seller": product.seller.is_platform_seller,
            "is_platform_fulfilled": product.seller.is_platform_fulfilled,
            "is_official_store": product.seller.is_official_store,
            "seller_source": _truncate(product.seller.seller_source, 80),
        }
    currency = normalize_currency_code(_truncate(product.currency, 16))
    price = product.price
    average_market_price = product.average_market_price
    extra_signals: list[str] = []
    if price is not None and not should_use_price_currency(
        currency,
        expected_currency=expected_currency,
        url=safe_url,
    ):
        if currency:
            extra_signals.append(f"price_ignored_localized_currency:{currency}")
        price = None
        currency = None
        average_market_price = None
    elif price is not None and currency is None:
        currency = expected_currency

    return ProductPageData(
        url=safe_url,
        site=_truncate(product.site, 255),
        product_title=_truncate(product.product_title, 240) or "Unknown product",
        description=_truncate(product.description, 1000),
        product_image_url=safe_image_url,
        price=price,
        currency=currency,
        average_market_price=average_market_price,
        seller=seller,
        return_policy=_truncate(clean_policy_snippet(product.return_policy), 1000),
        reviews=_sanitize_reviews(product.reviews),
        rating=_rating_or_none(product.rating),
        review_count=product.review_count,
        units_bought_recent=product.units_bought_recent,
        feedback_score=_feedback_score_or_none(product.feedback_score),
    ), extra_signals


def _feedback_score_or_none(value: int | None) -> int | None:
    if value is None:
        return None
    return max(0, min(100, value))


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


def _sanitize_reviews(reviews: list[ExtractedReviewInput]) -> list[ReviewInput]:
    cleaned_reviews: list[ReviewInput] = []
    seen: set[str] = set()
    for review in reviews[:50]:
        text = clean_review_body(review.text)
        if not text or text in seen:
            continue
        cleaned_reviews.append(
            ReviewInput(
                text=text[:2000],
                rating=_rating_or_none(review.rating),
                date=review.date,
                verified_purchase=review.verified_purchase,
            )
        )
        seen.add(text)
    return cleaned_reviews


def _rating_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    if value > 5:
        return None
    return value


def _signals_from_extracted_product(product: ProductPageData) -> list[str]:
    page_type = _page_type_from_url(product.url)
    signals = ["extension_dom", "title", f"page_type:{page_type}"]
    host = (product.site or urlparse(product.url).hostname or "").lower()
    if "amazon." in host:
        signals.append("site_amazon")
    elif host.endswith("ebay.com") or ".ebay." in host:
        signals.append("site_ebay")
    elif host.endswith("etsy.com") or ".etsy." in host:
        signals.append("site_etsy")
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


def _page_type_from_url(url: str) -> str:
    path = (urlparse(url).path or "").lower()
    if any(token in path for token in ("/cart", "/checkout", "/basket")):
        return "cart"
    if any(token in path for token in ("/account", "/signin", "/login")):
        return "account"
    if "/review" in path or "/reviews" in path:
        return "review_page"
    if "/search" in path:
        return "search"
    if any(token in path for token in ("/category", "/browse")):
        return "category"
    if any(token in path for token in ("/dp/", "/gp/product/", "/product", "/products/", "/item/", "/itm/")):
        return "product"
    if "bestbuy.com" in (urlparse(url).hostname or "") and path.endswith(".p"):
        return "product"
    return "unknown"


def _identity_confidence_from_signals(signals: list[str], page_type: str) -> float:
    if page_type in {"review_page", "search", "category", "cart", "account"}:
        return 0.0
    strong_signals = {
        "price",
        "seller",
        "rating_or_review_count",
        "review_text",
        "image",
        "policy",
    }
    count = len(strong_signals & set(signals))
    base = 0.25 if "title" in signals else 0.0
    return round(min(1.0, base + count * 0.15), 2)
