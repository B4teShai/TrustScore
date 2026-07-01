"""Server-side product-page fetch and extraction orchestration."""

from __future__ import annotations

from dataclasses import dataclass

from app.core.config import settings
from app.extraction.product_page import extract_product_page
from app.page_fetching.fetcher import (
    PageFetchError,
    URLValidationError,
    fetch_public_html,
    render_public_html,
)
from app.schemas.product_analysis import ProductPageData
from app.services.market_reference import enrich_market_reference


class ProductNotDetectedError(RuntimeError):
    """Raised when fetched content does not look like a product-detail page."""


@dataclass(frozen=True)
class ProductPageAnalysis:
    """Backend-extracted product data and extraction metadata."""

    product: ProductPageData
    fetch_mode: str
    signals: list[str]
    page_type: str = "product"
    product_identity_confidence: float = 1.0
    canonical_product_url: str | None = None


def analyze_product_url(
    url: str,
    *,
    target_market: str | None = "auto",
    locale: str | None = None,
) -> ProductPageAnalysis:
    """Fetch, optionally render, and extract product-page data."""
    static_error: PageFetchError | None = None
    static_result_reason = "This page does not have enough product-detail signals."
    render_url = url

    try:
        static_page = fetch_public_html(url)
    except PageFetchError as exc:
        static_page = None
        static_error = exc
    else:
        render_url = static_page.final_url
        static_result = extract_product_page(
            static_page.html,
            static_page.final_url,
            target_market=target_market,
            locale=locale,
        )
        static_result_reason = static_result.reason
        if static_result.detected and static_result.product is not None:
            product, market_signals = enrich_market_reference(
                static_result.product,
                target_market=target_market,
                locale=locale,
            )
            return ProductPageAnalysis(
                product=product,
                fetch_mode=static_page.mode,
                signals=[*static_result.signals, *market_signals],
                page_type=static_result.page_type,
                product_identity_confidence=static_result.product_identity_confidence,
                canonical_product_url=static_result.canonical_product_url,
            )

    rendered_page = None
    if settings.enable_rendered_fetch:
        try:
            rendered_page = render_public_html(render_url)
        except (PageFetchError, URLValidationError) as exc:
            if static_error is not None:
                raise PageFetchError(
                    f"Static fetch failed: {static_error} Rendered fallback failed: {exc}"
                ) from exc

    if rendered_page is not None:
        rendered_result = extract_product_page(
            rendered_page.html,
            rendered_page.final_url,
            target_market=target_market,
            locale=locale,
        )
        if rendered_result.detected and rendered_result.product is not None:
            product, market_signals = enrich_market_reference(
                rendered_result.product,
                target_market=target_market,
                locale=locale,
            )
            return ProductPageAnalysis(
                product=product,
                fetch_mode=rendered_page.mode,
                signals=[*rendered_result.signals, *market_signals],
                page_type=rendered_result.page_type,
                product_identity_confidence=rendered_result.product_identity_confidence,
                canonical_product_url=rendered_result.canonical_product_url,
            )

    if static_error is not None:
        raise PageFetchError(str(static_error)) from static_error
    raise ProductNotDetectedError(static_result_reason)
