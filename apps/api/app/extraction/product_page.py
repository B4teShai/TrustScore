"""Extract product analysis fields from fetched product-page HTML."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import ipaddress
import re
from typing import Any
from urllib.parse import urljoin, urlparse, urlunparse

from bs4 import BeautifulSoup

from app.schemas.product_analysis import ProductPageData, ReviewInput, SellerInfo
from app.services.market_context import (
    expected_currency_for_market,
    normalize_currency_code,
    resolve_target_market,
    should_use_price_currency,
)


PRICE_RE = re.compile(
    r"(US\s*\$|C\$|A\$|\$|USD|MNT|₮|€|EUR|£|GBP|¥|￥|円|JPY)\s?([0-9][0-9,.]*)",
    re.I,
)
PRICE_SUFFIX_RE = re.compile(
    r"([0-9][0-9,.]*)\s?(USD|MNT|EUR|GBP|JPY|円|CAD|AUD)",
    re.I,
)
PRICE_AMOUNT_TOKEN = r"(?:US\s*\$|C\$|A\$|\$|USD|MNT|₮|€|EUR|£|GBP|¥|￥|円|JPY)\s*[0-9][0-9,.]*"
AMAZON_FROM_PRICE_RE = re.compile(
    rf"(?:options?|offers?)\s+from\s*({PRICE_AMOUNT_TOKEN})",
    re.I,
)
AMAZON_DIRECT_PRICE_RE = re.compile(rf"({PRICE_AMOUNT_TOKEN})", re.I)
RATING_RE = re.compile(r"([0-5](?:\.[0-9])?)\s*(?:out of 5|stars?)", re.I)
REVIEW_COUNT_RE = re.compile(r"([0-9][0-9,]*)\s+(?:ratings?|reviews?)", re.I)
POLICY_RE = re.compile(
    r"(.{0,120}(?:return|refund|exchange|warranty|retour|remboursement|garantie|返品|返金|交換|保証).{0,220})",
    re.I,
)
PRODUCT_URL_RE = re.compile(r"/(dp|gp/product|product|products|item|itm)/", re.I)
BESTBUY_PRODUCT_URL_RE = re.compile(r"/site/[^/?#]+/\d+\.p(?:$|[?#])", re.I)
BOUGHT_RECENT_RE = re.compile(
    r"([0-9][0-9.,]*)\s*([KkMm])?\+?\s+bought\s+in\s+past\s+month", re.I
)
POLICY_ACCEPT_RE = re.compile(
    r"\b("
    r"return\s+policy|returns\s+policy|refund(?:s|ed|able)?|exchange|warranty|"
    r"replacement|returnable|eligible\s+for\s+returns?|free\s+returns?|return\s+window|"
    r"within\s+\d+\s*-?\s*(?:day|days|week|weeks|month|months)|"
    r"retour|remboursement|échange|echange|garantie|"
    r"返品|返金|交換|保証|返品無料"
    r")\b|(?:返品|返金|交換|保証|返品無料|\d+\s*日以内|30日)",
    re.I,
)
POLICY_TIME_RE = re.compile(
    r"("
    r"\b\d+\s*-?\s*(?:day|days|week|weeks|month|months)\b.{0,80}\breturns?\b|"
    r"\breturns?\b.{0,80}\b\d+\s*-?\s*(?:day|days|week|weeks|month|months)\b|"
    r"\b\d+\s*jours?\b.{0,80}\b(?:retour|remboursement|échange|echange)\b|"
    r"\b(?:retour|remboursement|échange|echange)\b.{0,80}\b\d+\s*jours?\b|"
    r"\d+\s*日以内|30日.{0,40}(?:返品|返金|交換)"
    r")",
    re.I,
)
POLICY_NAV_NOISE_RE = re.compile(
    r"\b(returns\s*&\s*orders|account\s*&\s*lists|hello,\s*sign in|today'?s deals|"
    r"prime video|gift cards|customer service|all\s+today'?s deals|cart\s+all)\b",
    re.I,
)
POLICY_START_RE = re.compile(
    r"\b(this item|item|eligible|free returns?|returns?|return policy|returns policy|refund|"
    r"exchange|warranty|replacement|returnable|retour|remboursement|garantie)\b|"
    r"(返品|返金|交換|保証|返品無料)",
    re.I,
)
REVIEW_BOILERPLATE_RE = re.compile(
    r"("
    r"brief content visible,\s*double tap to read full content\.?|"
    r"full content visible,\s*double tap to read brief content\.?|"
    r"read more\s+read less|"
    r"the media could not be loaded\.?"
    r")",
    re.I,
)


@dataclass(frozen=True)
class ExtractionResult:
    """Product extraction result plus detection metadata."""

    detected: bool
    product: ProductPageData | None
    reason: str
    signals: list[str] = field(default_factory=list)
    page_type: str = "unknown"
    product_identity_confidence: float = 0.0
    canonical_product_url: str | None = None


@dataclass(frozen=True)
class MarketplaceProfile:
    """Prioritized selectors for marketplace layouts when HTML is available."""

    site_signal: str | None = None
    title_selectors: tuple[str, ...] = ()
    image_selectors: tuple[str, ...] = ()
    price_selectors: tuple[str, ...] = ()
    seller_selectors: tuple[str, ...] = ()
    seller_reputation_selectors: tuple[str, ...] = ()
    rating_selectors: tuple[str, ...] = ()
    review_count_selectors: tuple[str, ...] = ()
    description_selectors: tuple[str, ...] = ()
    policy_selectors: tuple[str, ...] = ()
    review_card_selectors: tuple[str, ...] = ()
    review_body_selectors: tuple[str, ...] = ()


@dataclass(frozen=True)
class PriceExtraction:
    price: float | None
    currency: str | None
    ignored_currency: str | None = None


GENERIC_PROFILE = MarketplaceProfile(
    title_selectors=(
        "#productTitle",
        "[data-testid='product-title']",
        "[itemprop='name']",
        "h1",
        "meta[property='og:title']",
        "meta[name='twitter:title']",
        "title",
    ),
    image_selectors=(
        "meta[property='og:image']",
        "meta[name='twitter:image']",
        "[itemprop='image']",
        "img",
    ),
    price_selectors=(
        "meta[property='product:price:amount']",
        "[itemprop='price']",
        "[data-testid*='price']",
        "[class*='price']",
        "[id*='price']",
    ),
    seller_selectors=(
        "[data-testid*='seller']",
        "[class*='seller']",
        "[id*='seller']",
        "[class*='merchant']",
        "[id*='merchant']",
        "[itemprop='seller']",
        "[itemprop='brand']",
    ),
    rating_selectors=(
        "[itemprop='ratingValue']",
        "[aria-label*='out of 5']",
        "[aria-label*='stars']",
    ),
    review_count_selectors=(
        "[itemprop='reviewCount']",
        "[data-testid*='review-count']",
        "[aria-label*='review']",
    ),
    description_selectors=(
        "meta[name='description']",
        "meta[property='og:description']",
        "[itemprop='description']",
        "#productDescription",
        "[data-testid*='description']",
    ),
    policy_selectors=(
        "[data-testid*='return']",
        "[data-testid*='policy']",
        "[class*='return-policy']",
        "[id*='return-policy']",
        "[class*='returns-policy']",
        "[id*='returns-policy']",
    ),
    review_card_selectors=(
        "[itemprop='review']",
        "[data-testid*='review']",
        ".review",
    ),
    review_body_selectors=(
        "[itemprop='reviewBody']",
        "[data-review-text]",
        "p",
    ),
)


AMAZON_PROFILE = MarketplaceProfile(
    site_signal="site_amazon",
    title_selectors=(
        "#productTitle",
        "[data-feature-name='title'] h1",
        "meta[property='og:title']",
        "h1",
    ),
    image_selectors=(
        "#landingImage",
        "#imgTagWrapperId img",
        "meta[property='og:image']",
    ),
    price_selectors=(
        "#corePriceDisplay_desktop_feature_div .a-price .a-offscreen",
        "#apex_desktop .a-price .a-offscreen",
        "#corePrice_feature_div .a-price .a-offscreen",
        "#desktop_buybox .a-price .a-offscreen",
        "#buybox .a-price .a-offscreen",
        "#newBuyBoxPrice",
        "#priceblock_ourprice",
        "#priceblock_dealprice",
        "#price_inside_buybox",
        "meta[property='product:price:amount']",
    ),
    seller_selectors=(
        "#sellerProfileTriggerId",
        "[tabular-attribute-name='Sold by'] .tabular-buybox-text",
        "#merchant-info",
        "#bylineInfo",
    ),
    rating_selectors=("#acrPopover", "#averageCustomerReviews .a-icon-alt"),
    review_count_selectors=("#acrCustomerReviewText", "[data-hook='total-review-count']"),
    description_selectors=(
        "#feature-bullets",
        "#productDescription",
        "#bookDescription_feature_div",
        "meta[name='description']",
    ),
    policy_selectors=(
        "#productSupportAndReturnPolicy-secondary-content",
        "#productSupportAndReturnPolicy_feature_div",
        "#RETURNS_POLICY_feature_div",
        "#dp-returns-policy_feature_div",
        "[data-hook='returns-policy']",
        "#returns-policy-anchor-text",
        "[id*='RETURNS_POLICY']",
    ),
    review_card_selectors=(
        "[data-hook='review']",
        "[data-hook='cmps-review']",
        "div[id^='customer_review-']",
        "div[id^='customer_review_foreign-']",
    ),
    review_body_selectors=(
        "[data-hook='reviewText']",
        "[data-hook='reviewTextContainer']",
        "[data-hook='review-body']",
        "[data-hook='review-collapsed']",
        ".review-text-content",
        ".cr-original-review-text",
    ),
)


EBAY_PROFILE = MarketplaceProfile(
    site_signal="site_ebay",
    title_selectors=(
        "h1.x-item-title__mainTitle span",
        "[data-testid='x-item-title'] h1",
        "[data-testid='x-item-title']",
        "h1 span.ux-textspans",
        "meta[property='og:title']",
        "h1",
    ),
    image_selectors=(
        "img#icImg",
        "[data-testid='ux-image-carousel-item'] img",
        ".ux-image-carousel-item img",
        ".ux-image-carousel img",
        "meta[property='og:image']",
    ),
    price_selectors=(
        "[data-testid='x-price-primary'] span",
        ".x-price-primary span",
        ".x-price-approx span",
        "[itemprop='price']",
        "meta[property='product:price:amount']",
    ),
    seller_selectors=(
        "[data-testid='x-sellercard-atf__info'] a",
        ".x-sellercard-atf__info__about-seller a",
        "[data-testid='ux-seller-section__item'] a",
        "[class*='seller'] a",
    ),
    seller_reputation_selectors=(
        "[data-testid='x-sellercard-atf__data-item']",
        ".x-sellercard-atf__data-item",
        "[data-testid='ux-seller-section__item']",
    ),
    rating_selectors=("[data-testid*='rating']", "[aria-label*='out of 5 stars']"),
    review_count_selectors=(
        "[data-testid*='review-count']",
        "[aria-label*='review']",
        ".reviews",
    ),
    description_selectors=(
        "[data-testid='ux-layout-section__item']",
        "#viTabs_0_is",
        "meta[name='description']",
    ),
    policy_selectors=(
        "[data-testid='x-returns-minview']",
        "[data-testid='ux-labels-values-Returns']",
        "[data-testid='returns-policy']",
        "[id*='return-policy']",
        "[class*='return-policy']",
    ),
    review_card_selectors=(
        "[data-review-region]",
        "[data-review-text]",
        "[data-testid*='review']",
        ".review",
    ),
    review_body_selectors=("[data-review-text]", ".review-item-content", "p"),
)


ETSY_PROFILE = MarketplaceProfile(
    site_signal="site_etsy",
    title_selectors=(
        "h1[data-buy-box-listing-title]",
        "[data-buy-box-region='title'] h1",
        "h1.wt-text-body-01",
        "meta[property='og:title']",
        "h1",
    ),
    image_selectors=(
        "[data-carousel] img",
        ".listing-page-image-carousel-component img",
        "[data-listing-page-image] img",
        "meta[property='og:image']",
    ),
    price_selectors=(
        "[data-buy-box-region='price']",
        "p[data-buy-box-region='price']",
        "[data-selector='price-only']",
        "meta[property='product:price:amount']",
    ),
    seller_selectors=(
        "a[data-buy-box-region='shop-name']",
        "[data-buy-box-region='shop-name'] a",
        "a[data-region='shop-name']",
        "[data-region='shop-name'] a",
        "a[href*='/shop/'][data-region]",
        ".shop-name a",
        "a[href*='/shop/']",
    ),
    seller_reputation_selectors=(
        "[data-region='shop-rating']",
        "[aria-label*='shop reviews']",
        "a[href*='reviews']",
    ),
    rating_selectors=(
        "[data-review-star-rating]",
        "[aria-label*='out of 5 stars']",
        "[aria-label*='stars']",
    ),
    review_count_selectors=(
        "[data-reviews-total-count]",
        "a[href*='reviews']",
        "button[aria-label*='review']",
    ),
    description_selectors=(
        "[data-region='listing-page-description']",
        "[data-id='description-text']",
        "meta[name='description']",
    ),
    policy_selectors=(
        "[data-policies-return-policy]",
        "[data-region='listing-page-policies']",
        "[data-region='return-policy']",
        "[id*='return-policy']",
        "[class*='return-policy']",
    ),
    review_card_selectors=(
        "[data-review-region]",
        "[data-review-text]",
        "[data-testid*='review']",
        ".review",
    ),
    review_body_selectors=("[data-review-text]", "p"),
)


BESTBUY_PROFILE = MarketplaceProfile(
    site_signal="site_bestbuy",
    title_selectors=(
        "h1.heading-5",
        "h1[data-testid='product-title']",
        ".sku-title h1",
        "[data-testid='product-title']",
        "meta[property='og:title']",
    ),
    image_selectors=(
        "meta[property='og:image']",
        ".primary-image",
        "[data-testid='product-image'] img",
        "img[srcset]",
    ),
    price_selectors=(
        "[data-testid='customer-price']",
        ".priceView-customer-price span",
        ".pricing-price__regular-price",
        "[class*='priceView'] span",
        "meta[property='product:price:amount']",
    ),
    seller_selectors=(
        "[data-testid='brand-name']",
        ".vendor-display-name",
        "[class*='seller']",
        "[itemprop='brand']",
        "meta[property='product:brand']",
    ),
    rating_selectors=(
        "[data-testid='ratings-and-reviews']",
        "[aria-label*='out of 5']",
        ".c-ratings-reviews",
    ),
    review_count_selectors=(
        "[data-testid='ratings-and-reviews']",
        "a[href*='/site/reviews/']",
        "[aria-label*='reviews']",
    ),
    description_selectors=(
        "[data-testid='product-description']",
        ".shop-product-description",
        "meta[name='description']",
    ),
    policy_selectors=(
        "[data-testid*='return']",
        "[class*='fulfillment-return']",
        "[class*='return']",
        "[id*='return']",
    ),
    review_card_selectors=(
        "[data-testid*='review-card']",
        ".review-item",
        ".ugc-review",
        "[class*='review-item']",
    ),
    review_body_selectors=(
        "[data-testid*='review-text']",
        ".review-text",
        ".ugc-review-body",
        "p",
    ),
)


def extract_product_page(
    html: str,
    url: str,
    *,
    target_market: str | None = "auto",
    locale: str | None = None,
) -> ExtractionResult:
    """Extract one product-page payload from HTML."""
    soup = BeautifulSoup(html, "html.parser")
    page_text = soup.get_text(" ", strip=True)
    profile = _profile_for_url(url)
    page_type = _page_type_for_url(url)
    canonical_product_url = _canonical_product_url_from_page(soup, url)
    resolved_market = resolve_target_market(target_market, url=url, locale=locale)
    expected_currency = expected_currency_for_market(resolved_market)
    product_objects = _json_ld_products(soup)
    title = _clean_title(_first_product_value(product_objects, "name") or _find_title(soup, profile))
    site = urlparse(url).hostname

    if not title:
        return ExtractionResult(
            detected=False,
            product=None,
            reason="No clear product title was found.",
            page_type=page_type,
            canonical_product_url=canonical_product_url,
        )

    price_result = _extract_price_and_currency(
        soup,
        page_text,
        product_objects,
        profile,
        expected_currency=expected_currency,
        url=url,
    )
    seller = _extract_seller(soup, page_text, product_objects, profile)
    reviews = _extract_reviews(soup, product_objects, profile)
    product = ProductPageData(
        url=canonical_product_url or url,
        site=site,
        product_title=title,
        description=_extract_description(soup, product_objects, profile),
        product_image_url=_extract_image(soup, url, product_objects, profile),
        price=price_result.price,
        currency=price_result.currency,
        average_market_price=None,
        seller=seller,
        return_policy=_extract_policy(soup, page_text, profile),
        reviews=reviews,
        rating=_extract_rating(soup, page_text, product_objects, profile),
        review_count=_extract_review_count(soup, page_text, product_objects, profile),
        units_bought_recent=_extract_units_bought_recent(page_text),
    )
    signals = _product_signals(
        product,
        page_text,
        bool(product_objects),
        url,
        page_type=page_type,
        canonical_product_url=canonical_product_url,
    )
    if price_result.ignored_currency:
        signals.append(f"price_ignored_localized_currency:{price_result.ignored_currency}")
    identity_confidence = _product_identity_confidence(signals)
    detected = _is_product_detail(signals)
    return ExtractionResult(
        detected=detected,
        product=product if detected else None,
        reason=(
            "Product page signals found."
            if detected
            else "This page does not have enough product-detail signals."
        ),
        signals=signals,
        page_type=page_type,
        product_identity_confidence=identity_confidence,
        canonical_product_url=canonical_product_url,
    )


def _json_ld_products(soup: BeautifulSoup) -> list[dict[str, Any]]:
    products: list[dict[str, Any]] = []
    for script in soup.select("script[type='application/ld+json']"):
        text = script.string or script.get_text() or ""
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            continue
        products.extend(_find_product_objects(parsed))
    return products


def _find_product_objects(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [product for item in value for product in _find_product_objects(item)]
    if not isinstance(value, dict):
        return []

    raw_type = value.get("@type")
    type_values = raw_type if isinstance(raw_type, list) else [raw_type]
    nested = [product for item in value.values() for product in _find_product_objects(item)]
    if any(str(item).lower() == "product" for item in type_values):
        return [value, *nested]
    return nested


def _first_product_value(products: list[dict[str, Any]], key: str) -> str | None:
    for product in products:
        value = _string_from_unknown(product.get(key))
        if value:
            return value
    return None


def _profile_for_url(url: str) -> MarketplaceProfile:
    host = (urlparse(url).hostname or "").lower()
    if "amazon." in host:
        return AMAZON_PROFILE
    if host.endswith("ebay.com") or ".ebay." in host:
        return EBAY_PROFILE
    if host.endswith("etsy.com") or ".etsy." in host:
        return ETSY_PROFILE
    if host.endswith("bestbuy.com") or ".bestbuy." in host:
        return BESTBUY_PROFILE
    return GENERIC_PROFILE


def _find_title(soup: BeautifulSoup, profile: MarketplaceProfile) -> str | None:
    return _first_selector_text(soup, (*profile.title_selectors, *GENERIC_PROFILE.title_selectors))


def _clean_title(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = re.sub(r"\s+", " ", value).strip()
    cleaned = re.sub(r"\s+[-|]\s+Amazon(?:\.com)?$", "", cleaned, flags=re.I)
    if len(cleaned) < 4:
        return None
    return _truncate_at_word(cleaned, 240)


def _extract_description(
    soup: BeautifulSoup,
    products: list[dict[str, Any]],
    profile: MarketplaceProfile,
) -> str | None:
    value = _first_product_value(products, "description")
    if value:
        return _compact(value)[:500]
    for selector in (*profile.description_selectors, *GENERIC_PROFILE.description_selectors):
        value = _element_text(soup.select_one(selector))
        if value:
            return _compact(value)[:500]
    return None


def _extract_image(
    soup: BeautifulSoup,
    page_url: str,
    products: list[dict[str, Any]],
    profile: MarketplaceProfile,
) -> str | None:
    for product in products:
        image_url = _image_from_unknown(product.get("image"), page_url)
        if image_url:
            return image_url

    for selector in (*profile.image_selectors, *GENERIC_PROFILE.image_selectors):
        element = soup.select_one(selector)
        image_url = _image_from_element(element, page_url)
        if image_url:
            return image_url
    return None


def _extract_price_and_currency(
    soup: BeautifulSoup,
    page_text: str,
    products: list[dict[str, Any]],
    profile: MarketplaceProfile,
    *,
    expected_currency: str,
    url: str,
) -> PriceExtraction:
    candidates: list[tuple[float, str | None]] = []

    for product in products:
        for offer in _as_list(product.get("offers")):
            if not isinstance(offer, dict):
                continue
            price = _parse_float(offer.get("price"))
            if price is not None:
                currency = normalize_currency_code(_string_from_unknown(offer.get("priceCurrency")))
                candidates.append((price, currency))

    page_currency = normalize_currency_code(
        _element_text(soup.select_one("meta[property='product:price:currency']"))
    )
    if profile.site_signal == "site_amazon":
        for value in _amazon_price_texts(soup):
            price, parsed_currency = _parse_price_text(value, allow_without_currency=False)
            if price is not None:
                candidates.append((price, parsed_currency or page_currency))

    for selector in (*profile.price_selectors, *GENERIC_PROFILE.price_selectors):
        value = _element_text(soup.select_one(selector))
        price, parsed_currency = _parse_price_text(value, allow_without_currency=True)
        if price is not None:
            candidates.append((price, parsed_currency or page_currency))

    for value in _split_price_texts(soup):
        price, parsed_currency = _parse_price_text(value, allow_without_currency=True)
        if price is not None:
            candidates.append((price, parsed_currency or page_currency))

    price, parsed_currency = _parse_price_text(page_text, allow_without_currency=False)
    if price is not None:
        candidates.append((price, parsed_currency))

    return _select_price_candidate(candidates, expected_currency=expected_currency, url=url)


def _split_price_texts(soup: BeautifulSoup) -> list[str]:
    values: list[str] = []
    containers = soup.select(
        "#corePriceDisplay_desktop_feature_div, #apex_desktop, #corePrice_feature_div, #desktop_buybox, #buybox"
    )
    for container in containers:
        symbol = _element_text(container.select_one(".a-price-symbol"))
        whole = _element_text(container.select_one(".a-price-whole"))
        fraction = _element_text(container.select_one(".a-price-fraction"))
        if not whole:
            continue
        compact_whole = whole.replace(" ", "").rstrip(".,")
        if symbol and fraction:
            values.append(f"{symbol}{compact_whole}.{fraction}")
        elif symbol:
            values.append(f"{symbol}{compact_whole}")
        elif fraction:
            values.append(f"{compact_whole}.{fraction}")
        else:
            values.append(compact_whole)
    return values


def _amazon_price_texts(soup: BeautifulSoup) -> list[str]:
    values: list[str] = []
    for selector in (
        "#desktop_buybox",
        "#buybox",
        "#centerCol",
        "#twister",
        "#variation_color_name",
        "[data-feature-name='twister']",
    ):
        value = _price_phrase_from_text(_element_text(soup.select_one(selector)))
        if value and value not in values:
            values.append(value)
    return values


def _price_phrase_from_text(value: str | None) -> str | None:
    if not value:
        return None
    match = AMAZON_FROM_PRICE_RE.search(value)
    if match:
        return match.group(1)
    match = AMAZON_DIRECT_PRICE_RE.search(value)
    return match.group(1) if match else None


def _extract_rating(
    soup: BeautifulSoup,
    page_text: str,
    products: list[dict[str, Any]],
    profile: MarketplaceProfile,
) -> float | None:
    for product in products:
        aggregate = product.get("aggregateRating")
        if isinstance(aggregate, dict):
            rating = _parse_float(aggregate.get("ratingValue"))
            if rating is not None and 0 <= rating <= 5:
                return rating

    for selector in (*profile.rating_selectors, *GENERIC_PROFILE.rating_selectors):
        rating = _parse_rating_text(_element_text(soup.select_one(selector)))
        if rating is not None:
            return rating

    match = RATING_RE.search(page_text)
    if not match:
        return None
    rating = _parse_float(match.group(1))
    return rating if rating is not None and 0 <= rating <= 5 else None


def _extract_review_count(
    soup: BeautifulSoup,
    page_text: str,
    products: list[dict[str, Any]],
    profile: MarketplaceProfile,
) -> int | None:
    for product in products:
        aggregate = product.get("aggregateRating")
        if isinstance(aggregate, dict):
            count = _parse_int(aggregate.get("reviewCount") or aggregate.get("ratingCount"))
            if count is not None:
                return count

    for selector in (*profile.review_count_selectors, *GENERIC_PROFILE.review_count_selectors):
        element = soup.select_one(selector)
        text_value = _element_text(element)
        count_match = REVIEW_COUNT_RE.search(text_value or "")
        count = _parse_int(count_match.group(1)) if count_match else _parse_int(text_value)
        if count is None and element is not None:
            for attribute in ("content", "data-reviews-total-count", "aria-label"):
                attribute_value = element.get(attribute)
                count_match = REVIEW_COUNT_RE.search(attribute_value or "")
                count = _parse_int(count_match.group(1)) if count_match else _parse_int(attribute_value)
                if count is not None:
                    break
        if count is not None:
            return count

    match = REVIEW_COUNT_RE.search(page_text)
    return _parse_int(match.group(1)) if match else None


def _extract_seller(
    soup: BeautifulSoup,
    page_text: str,
    products: list[dict[str, Any]],
    profile: MarketplaceProfile,
) -> SellerInfo | None:
    name: str | None = None
    source: str | None = None
    for product in products:
        seller = _seller_from_unknown(product.get("seller") or product.get("brand"))
        if seller:
            name = seller
            source = "structured"
            break
        for offer in _as_list(product.get("offers")):
            seller = _seller_from_unknown(offer.get("seller") if isinstance(offer, dict) else None)
            if seller:
                name = seller
                source = "structured_offer"
                break
        if name:
            break

    if not name:
        for selector in (*profile.seller_selectors, *GENERIC_PROFILE.seller_selectors):
            seller = _clean_seller(_element_text(soup.select_one(selector)))
            if seller:
                name = seller
                source = "byline" if "byline" in selector.lower() else "selector"
                break
    if not name:
        name = _clean_seller(page_text)
        source = "page_text" if name else None

    reputation_text = " ".join(
        _selector_texts(
            soup,
            (*profile.seller_reputation_selectors, *GENERIC_PROFILE.seller_selectors),
            limit=8,
        )
    )
    seller_rating, seller_review_count = _extract_seller_reputation(reputation_text)
    if not name and seller_rating is None and seller_review_count is None:
        return None
    sold_by = _extract_labeled_party(page_text, "Sold by")
    ships_from = _extract_labeled_party(page_text, "Ships from")
    fulfilled_by = _extract_labeled_party(page_text, "Fulfilled by")
    is_platform = _is_platform_party(name) or _is_platform_party(sold_by)
    is_fulfilled = _is_platform_party(fulfilled_by) or _is_platform_party(ships_from)
    is_official = _looks_like_official_store(name)
    return SellerInfo(
        name=name,
        rating=seller_rating,
        review_count=seller_review_count,
        sold_by=sold_by,
        ships_from=ships_from,
        fulfilled_by=fulfilled_by,
        brand_store_name=name if is_official else None,
        is_platform_seller=is_platform or None,
        is_platform_fulfilled=is_fulfilled or None,
        is_official_store=is_official or None,
        seller_source=source,
    )


def _extract_labeled_party(page_text: str, label: str) -> str | None:
    match = re.search(rf"\b{re.escape(label)}\s+(.+?)(?:\s+Ships from|\s+Sold by|\s+Fulfilled by|\s*$)", page_text, re.I)
    if not match:
        return None
    return _clean_seller(match.group(1))


def _is_platform_party(value: str | None) -> bool:
    if not value:
        return False
    return bool(re.search(r"\b(Amazon|Best Buy|Walmart|Target|Etsy|eBay)\b", value, re.I))


def _looks_like_official_store(value: str | None) -> bool:
    if not value:
        return False
    return bool(
        re.search(
            r"\b(Apple|Amazon|Logitech|StarTech|Sony|Samsung|Microsoft|Anker|Belkin|Dell|HP)\b.*\b(Store|Official)\b",
            value,
            re.I,
        )
    )


def _extract_policy(
    soup: BeautifulSoup,
    page_text: str,
    profile: MarketplaceProfile,
) -> str | None:
    for selector in (*profile.policy_selectors, *GENERIC_PROFILE.policy_selectors):
        policy = clean_policy_snippet(_element_text(soup.select_one(selector)))
        if policy:
            return policy

    for element in soup.find_all(("p", "li", "div", "span")):
        policy = clean_policy_snippet(_element_text(element))
        if policy:
            return policy

    match = POLICY_RE.search(page_text)
    if not match:
        return None
    for match in POLICY_RE.finditer(page_text):
        policy = clean_policy_snippet(match.group(1))
        if policy:
            return policy
    return None


def clean_policy_snippet(value: str | None) -> str | None:
    """Return a compact policy snippet only when it contains real policy terms."""
    if not value:
        return None
    compact = _compact(value)
    if not compact:
        return None
    if not (POLICY_ACCEPT_RE.search(compact) or POLICY_TIME_RE.search(compact)):
        return None
    compact = _focus_policy_snippet(compact)
    if POLICY_NAV_NOISE_RE.search(compact) and not _has_strong_policy_detail(compact):
        return None
    return compact[:1000]


def _focus_policy_snippet(value: str) -> str:
    sentence_end = re.search(r"(?<=[.!?])\s+", value)
    if len(value) <= 180 and _has_strong_policy_detail(value) and sentence_end is None:
        return value
    anchor = POLICY_TIME_RE.search(value) or POLICY_ACCEPT_RE.search(value)
    if not anchor:
        return value
    context_start = max(0, anchor.start() - 120)
    context_end = min(len(value), anchor.end() + 220)
    context = value[context_start:context_end].strip()
    start_matches = list(POLICY_START_RE.finditer(context[: max(anchor.end() - context_start, 0)]))
    if start_matches and not re.match(r"^\d", context):
        preferred = next(
            (match for match in start_matches if match.group(1).lower() == "this item"),
            start_matches[-1],
        )
        context = context[preferred.start() :].strip()
    sentence_end = re.search(r"(?<=[.!?])\s+", context)
    if sentence_end:
        return context[: sentence_end.end()].strip()
    return context


def _has_strong_policy_detail(value: str) -> bool:
    return bool(
        re.search(
            r"\b(refund|exchange|warranty|replacement|returnable|eligible\s+for\s+returns?|"
            r"free\s+returns?|return\s+policy|returns\s+policy|retour|remboursement|"
            r"échange|echange|garantie)\b|(?:返品|返金|交換|保証|返品無料|30日|\d+\s*日以内)",
            value,
            flags=re.I,
        )
        or POLICY_TIME_RE.search(value)
    )


def clean_review_body(value: str | None) -> str | None:
    """Remove marketplace UI boilerplate from visible review text."""
    if not value:
        return None
    cleaned = REVIEW_BOILERPLATE_RE.sub(" ", value)
    cleaned = _compact(cleaned)
    return cleaned or None


def _extract_reviews(
    soup: BeautifulSoup,
    products: list[dict[str, Any]],
    profile: MarketplaceProfile,
) -> list[ReviewInput]:
    reviews: list[ReviewInput] = []
    seen: set[str] = set()
    for product in products:
        for raw_review in _as_list(product.get("review")):
            if not isinstance(raw_review, dict):
                continue
            text = clean_review_body(
                _string_from_unknown(raw_review.get("reviewBody") or raw_review.get("description")) or ""
            )
            if not text or text in seen:
                continue
            rating = None
            review_rating = raw_review.get("reviewRating")
            if isinstance(review_rating, dict):
                rating = _parse_float(review_rating.get("ratingValue"))
            reviews.append(ReviewInput(text=text[:1000], rating=rating))
            seen.add(text)
            if len(reviews) >= 10:
                return reviews

    for card in _select_review_cards(soup, profile):
        text = clean_review_body(_review_body_from_card(card, profile))
        if not text or len(text) < 20 or text in seen:
            continue
        rating = _parse_rating_text(_element_text(card))
        reviews.append(ReviewInput(text=text[:1000], rating=rating))
        seen.add(text)
        if len(reviews) >= 10:
            return reviews

    # Current Amazon layout: each card is data-hook="review" with the body in
    # data-hook="reviewText". Prefer that precise body over the whole card text.
    for card in soup.select("[data-hook='review'], [data-hook='cmps-review']"):
        body = card.select_one("[data-hook='reviewText'], [data-hook='reviewTextContainer']")
        text = clean_review_body((body or card).get_text(" ", strip=True))
        if not text or len(text) < 20 or text in seen:
            continue
        reviews.append(ReviewInput(text=text[:1000]))
        seen.add(text)
        if len(reviews) >= 10:
            return reviews

    for element in soup.find_all(
        lambda tag: any(
            "review" in str(value).lower()
            for value in [
                tag.get("id"),
                " ".join(tag.get("class", [])),
                tag.get("data-testid"),
                tag.get("data-hook"),
            ]
            if value
        )
    ):
        text = clean_review_body(element.get_text(" ", strip=True))
        if not text or len(text) < 20 or text in seen:
            continue
        reviews.append(ReviewInput(text=text[:1000]))
        seen.add(text)
        if len(reviews) >= 10:
            break
    return reviews


def _extract_units_bought_recent(page_text: str) -> int | None:
    match = BOUGHT_RECENT_RE.search(page_text)
    if not match:
        return None
    base = _parse_float(match.group(1))
    if base is None:
        return None
    unit = (match.group(2) or "").lower()
    scale = 1000 if unit == "k" else 1_000_000 if unit == "m" else 1
    return int(round(base * scale))


def _product_signals(
    product: ProductPageData,
    page_text: str,
    has_structured_product: bool,
    url: str,
    *,
    page_type: str,
    canonical_product_url: str | None,
) -> list[str]:
    signals: list[str] = []
    if has_structured_product:
        signals.append("structured_product")
    signals.append(f"page_type:{page_type}")
    if canonical_product_url:
        signals.append("canonical_product_url_found")
    site_signal = _site_signal(url)
    if site_signal:
        signals.append(site_signal)
    if product.product_title:
        signals.append("title")
    if product.price is not None:
        signals.append("price")
    if product.rating is not None or product.review_count is not None:
        signals.append("rating_or_review_count")
    if product.reviews:
        signals.append("review_text")
    if product.seller and product.seller.name:
        signals.append("seller")
    if product.product_image_url:
        signals.append("image")
    if product.return_policy:
        signals.append("policy")
    elif POLICY_NAV_NOISE_RE.search(page_text):
        signals.append("policy_rejected_noisy_text")
    if re.search(r"\b(add to cart|add to bag|buy now|checkout|in stock)\b", page_text, re.I):
        signals.append("commerce_action")
    if PRODUCT_URL_RE.search(urlparse(url).path) or BESTBUY_PRODUCT_URL_RE.search(urlparse(url).path):
        signals.append("product_url")
    return signals


def _is_product_detail(signals: list[str]) -> bool:
    has = signals.__contains__
    page_type = _signal_value(signals, "page_type:") or "unknown"
    if page_type in {"review_page", "search", "category", "cart", "account"} and not (
        has("structured_product") or has("canonical_product_url_found")
    ):
        return False
    commerce = (
        has("price")
        or has("commerce_action")
        or has("rating_or_review_count")
        or has("review_text")
        or has("seller")
    )
    if has("structured_product") and commerce:
        return True
    if has("title") and has("price") and (has("seller") or has("rating_or_review_count") or has("commerce_action")):
        return True
    product_signal_count = sum(
        1
        for signal in (
            "price",
            "seller",
            "rating_or_review_count",
            "review_text",
            "image",
            "commerce_action",
            "structured_product",
            "product_url",
        )
        if has(signal)
    )
    if product_signal_count < 2:
        return False
    score = sum(
        {
            "structured_product": 3,
            "title": 2,
            "price": 2,
            "rating_or_review_count": 2,
            "review_text": 2,
            "seller": 1,
            "image": 1,
            "policy": 1,
            "commerce_action": 2,
            "product_url": 1,
        }.get(signal, 0)
        for signal in signals
    )
    return score >= 6 and has("title") and commerce


def _product_identity_confidence(signals: list[str]) -> float:
    if not _is_product_detail(signals):
        return 0.0
    has = signals.__contains__
    score = 0.15 if has("title") else 0.0
    for signal, weight in (
        ("structured_product", 0.25),
        ("product_url", 0.15),
        ("canonical_product_url_found", 0.10),
        ("price", 0.15),
        ("seller", 0.12),
        ("rating_or_review_count", 0.10),
        ("image", 0.08),
        ("commerce_action", 0.05),
    ):
        if has(signal):
            score += weight
    page_type = _signal_value(signals, "page_type:") or "unknown"
    if page_type != "product":
        score = min(score, 0.65)
    return round(max(0.0, min(1.0, score)), 2)


def _signal_value(signals: list[str], prefix: str) -> str | None:
    for signal in signals:
        if signal.startswith(prefix):
            return signal.removeprefix(prefix)
    return None


def _site_signal(url: str) -> str | None:
    host = (urlparse(url).hostname or "").lower()
    if "amazon." in host:
        return "site_amazon"
    if host.endswith("ebay.com") or ".ebay." in host:
        return "site_ebay"
    if host.endswith("etsy.com") or ".etsy." in host:
        return "site_etsy"
    if host.endswith("bestbuy.com") or ".bestbuy." in host:
        return "site_bestbuy"
    return None


def _page_type_for_url(url: str) -> str:
    path = (urlparse(url).path or "").lower()
    if re.search(r"/(cart|checkout|basket|account|signin|login)(?:/|$)", path):
        return "cart" if "cart" in path or "basket" in path or "checkout" in path else "account"
    if re.search(r"/(search|s)(?:/|$)", path) or "search" in path:
        return "search"
    if re.search(r"/(category|categories|browse|b)(?:/|$)", path):
        return "category"
    if re.search(r"/reviews?(?:/|$)", path):
        return "review_page"
    if PRODUCT_URL_RE.search(path) or BESTBUY_PRODUCT_URL_RE.search(path):
        return "product"
    return "unknown"


def _canonical_product_url_from_page(soup: BeautifulSoup, page_url: str) -> str | None:
    host = (urlparse(page_url).hostname or "").lower()
    if not (host.endswith("bestbuy.com") or ".bestbuy." in host):
        return None
    for selector in ("link[rel='canonical']", "a[href*='/site/'][href*='.p']", "meta[property='og:url']"):
        element = soup.select_one(selector)
        raw = None
        if element is not None:
            raw = element.get("href") or element.get("content")
        if not raw:
            continue
        candidate = urljoin(page_url, raw)
        parsed = urlparse(candidate)
        if (parsed.hostname or "").lower().endswith("bestbuy.com") and BESTBUY_PRODUCT_URL_RE.search(parsed.path):
            return urlunparse(parsed._replace(fragment=""))
    return None


def _element_text(element: Any) -> str | None:
    if element is None:
        return None
    if getattr(element, "name", "") == "meta":
        return _compact(element.get("content") or "")
    return _compact(element.get_text(" ", strip=True))


def _first_selector_text(soup: BeautifulSoup, selectors: tuple[str, ...]) -> str | None:
    values = _selector_texts(soup, selectors, limit=1)
    return values[0] if values else None


def _selector_texts(
    soup: BeautifulSoup,
    selectors: tuple[str, ...],
    *,
    limit: int,
) -> list[str]:
    values: list[str] = []
    seen_selectors: set[str] = set()
    seen_values: set[str] = set()
    seen: set[str] = set()
    for selector in selectors:
        if selector in seen_selectors:
            continue
        seen_selectors.add(selector)
        for element in soup.select(selector):
            marker = str(id(element))
            if marker in seen:
                continue
            seen.add(marker)
            value = _element_text(element)
            if not value or value in seen_values:
                continue
            values.append(value)
            seen_values.add(value)
            if len(values) >= limit:
                return values
    return values


def _select_review_cards(soup: BeautifulSoup, profile: MarketplaceProfile) -> list[Any]:
    cards: list[Any] = []
    seen: set[int] = set()
    for selector in (*profile.review_card_selectors, *GENERIC_PROFILE.review_card_selectors):
        for card in soup.select(selector):
            marker = id(card)
            if marker in seen:
                continue
            seen.add(marker)
            cards.append(card)
    return cards


def _review_body_from_card(card: Any, profile: MarketplaceProfile) -> str | None:
    for selector in (*profile.review_body_selectors, *GENERIC_PROFILE.review_body_selectors):
        body = card.select_one(selector)
        if body is not None:
            value = _element_text(body)
            if value:
                return value
    return _element_text(card)


def _compact(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _truncate_at_word(value: str, max_length: int) -> str:
    if len(value) <= max_length:
        return value
    boundary = value.rfind(" ", 0, max_length + 1)
    if boundary < max_length * 0.65:
        return value[:max_length].rstrip()
    return value[:boundary].rstrip()


def _absolute_http_url(value: str | None, base_url: str) -> str | None:
    if not value:
        return None
    try:
        candidate = urljoin(base_url, value.replace("&amp;", "&").strip())
        return _validate_public_reference_url(candidate)
    except ValueError:
        return None


def _validate_public_reference_url(raw_url: str) -> str:
    """Validate a URL that is returned to the extension but not fetched server-side."""
    parsed = urlparse(raw_url.strip())
    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError("Only HTTP and HTTPS URLs are supported.")
    if not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("URL must include a safe hostname.")

    hostname = parsed.hostname.strip().lower().rstrip(".")
    if (
        hostname in {"localhost", "localhost.localdomain"}
        or hostname.endswith(".localhost")
        or hostname.endswith(".local")
    ):
        raise ValueError("Local URLs are not supported.")

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
        raise ValueError("Private network URLs are not supported.")

    normalized = parsed._replace(
        scheme=parsed.scheme.lower(),
        netloc=hostname if parsed.port is None else f"{hostname}:{parsed.port}",
        fragment="",
    )
    return urlunparse(normalized)


def _image_from_element(element: Any, base_url: str) -> str | None:
    if element is None:
        return None

    for attribute in ("content", "src", "data-src", "data-old-hires"):
        image_url = _absolute_http_url(element.get(attribute), base_url)
        if image_url:
            return image_url

    for attribute in ("srcset", "data-srcset"):
        image_url = _image_from_srcset(element.get(attribute), base_url)
        if image_url:
            return image_url

    for attribute in ("data-a-dynamic-image", "data-dynamic-image"):
        image_url = _image_from_dynamic_json(element.get(attribute), base_url)
        if image_url:
            return image_url

    return None


def _image_from_srcset(value: str | None, base_url: str) -> str | None:
    if not value:
        return None

    candidates: list[tuple[float, str]] = []
    for raw_candidate in value.split(","):
        parts = raw_candidate.strip().split()
        if not parts:
            continue
        image_url = _absolute_http_url(parts[0], base_url)
        if not image_url:
            continue
        score = 1.0
        if len(parts) > 1:
            descriptor = parts[1]
            try:
                score = float(descriptor.rstrip("wx"))
            except ValueError:
                score = 1.0
        candidates.append((score, image_url))

    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _image_from_dynamic_json(value: str | None, base_url: str) -> str | None:
    if not value:
        return None

    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, dict):
        return None

    candidates: list[tuple[float, str]] = []
    for raw_url, size in parsed.items():
        image_url = _absolute_http_url(str(raw_url), base_url)
        if not image_url:
            continue
        area = 0.0
        if isinstance(size, list) and len(size) >= 2:
            try:
                area = float(size[0]) * float(size[1])
            except (TypeError, ValueError):
                area = 0.0
        candidates.append((area, image_url))

    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _image_from_unknown(value: Any, base_url: str) -> str | None:
    if isinstance(value, str):
        return _absolute_http_url(value, base_url)
    if isinstance(value, list):
        for item in value:
            image_url = _image_from_unknown(item, base_url)
            if image_url:
                return image_url
    if isinstance(value, dict):
        return (
            _image_from_unknown(value.get("url"), base_url)
            or _image_from_unknown(value.get("contentUrl"), base_url)
            or _image_from_unknown(value.get("thumbnailUrl"), base_url)
        )
    return None


def _string_from_unknown(value: Any) -> str | None:
    if isinstance(value, str):
        return _compact(value)
    if isinstance(value, list):
        for item in value:
            result = _string_from_unknown(item)
            if result:
                return result
    if isinstance(value, dict):
        return _string_from_unknown(value.get("name") or value.get("url"))
    return None


def _seller_from_unknown(value: Any) -> str | None:
    return _clean_seller(_string_from_unknown(value))


def _clean_seller(value: str | None) -> str | None:
    if not value:
        return None
    compact = _compact(value)
    patterns = [
        r"\bSold by\s+(.+?)(?:\s+and\s+Fulfilled|\s+Ships from|\s*$)",
        r"\bSeller[:\s]+(.+?)(?:\s{2,}|$)",
        r"\bShips from\s+(.+?)(?:\s+Sold by|\s*$)",
        r"\bVisit the\s+(.+?\s+Store)\b",
        r"\bBrand[:\s]+(.+?)(?:\s{2,}|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, compact, flags=re.I)
        if match:
            cleaned = re.sub(r"[|•].*$", "", match.group(1)).strip()
            return cleaned if 2 <= len(cleaned) <= 80 else None
    cleaned = re.sub(r"^(Sold by|Seller:|Visit the)\s+", "", compact, flags=re.I)
    cleaned = re.sub(r"^Shop\s+", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s+(?:on Etsy|\|\s*eBay)$", "", cleaned, flags=re.I)
    cleaned = re.sub(r"[|•].*$", "", cleaned).strip()
    return cleaned if 2 <= len(cleaned) <= 80 else None


def _extract_seller_reputation(value: str | None) -> tuple[float | None, int | None]:
    if not value:
        return None, None
    rating = _rating_from_percent(value) or _parse_rating_text(value)
    review_count = None
    match = re.search(
        r"([0-9][0-9,]*)\s*(?:feedback|seller reviews?|shop reviews?|reviews?)",
        value,
        flags=re.I,
    )
    if match:
        review_count = _parse_int(match.group(1))
    return rating, review_count


def _rating_from_percent(value: str | None) -> float | None:
    if not value:
        return None
    match = re.search(r"([0-9]{1,3}(?:[.,][0-9]+)?)\s*%\s*(?:positive|feedback)?", value, flags=re.I)
    if not match:
        return None
    percent = _parse_float(match.group(1))
    if percent is None or percent > 100:
        return None
    return round(percent / 20, 1)


def _parse_rating_text(value: str | None) -> float | None:
    if not value:
        return None
    match = RATING_RE.search(value)
    if match:
        rating = _parse_float(match.group(1))
        return rating if rating is not None and 0 <= rating <= 5 else None
    number_match = re.search(r"([0-5](?:[.,][0-9])?)", value)
    if not number_match:
        return None
    rating = _parse_float(number_match.group(1))
    return rating if rating is not None and 0 <= rating <= 5 else None


def _parse_price_text(
    value: str | None,
    *,
    allow_without_currency: bool,
) -> tuple[float | None, str | None]:
    if not value:
        return None, None
    compact = _compact(value)
    match = PRICE_RE.search(compact)
    if match:
        currency = _currency_from_token(match.group(1))
        return _parse_amount(match.group(2), currency), currency

    suffix_match = PRICE_SUFFIX_RE.search(compact)
    if suffix_match:
        currency = _currency_from_token(suffix_match.group(2))
        return _parse_amount(suffix_match.group(1), currency), currency

    if not allow_without_currency:
        return None, None
    amount_match = re.search(r"([0-9][0-9,.]*[0-9]|[0-9])", compact)
    if not amount_match:
        return None, None
    return _parse_amount(amount_match.group(1), None), None


def _select_price_candidate(
    candidates: list[tuple[float, str | None]],
    *,
    expected_currency: str,
    url: str,
) -> PriceExtraction:
    if not candidates:
        return PriceExtraction(price=None, currency=None)

    normalized_candidates = [
        (price, normalize_currency_code(currency))
        for price, currency in candidates
        if price is not None
    ]
    for price, currency in normalized_candidates:
        if currency == expected_currency:
            return PriceExtraction(price=price, currency=currency)

    for price, currency in normalized_candidates:
        if currency is None:
            return PriceExtraction(price=price, currency=expected_currency)

    ignored_currency = None
    for price, currency in normalized_candidates:
        if should_use_price_currency(currency, expected_currency=expected_currency, url=url):
            return PriceExtraction(price=price, currency=currency)
        if ignored_currency is None and currency:
            ignored_currency = currency

    return PriceExtraction(price=None, currency=None, ignored_currency=ignored_currency)


def _currency_from_token(value: str | None) -> str | None:
    return normalize_currency_code(value)


def _parse_amount(value: str | None, currency: str | None) -> float | None:
    if not value:
        return None
    raw = re.sub(r"[^\d.,]", "", value)
    if not raw:
        return None
    if currency and currency.upper() in {"JPY", "KRW", "VND", "CLP", "ISK"}:
        return _parse_float(raw.replace(",", "").replace(".", ""))
    decimal_index = max(raw.rfind("."), raw.rfind(","))
    trailing = len(raw) - decimal_index - 1 if decimal_index >= 0 else 0
    if decimal_index >= 0 and 1 <= trailing <= 2:
        integer_part = re.sub(r"[.,]", "", raw[:decimal_index])
        return _parse_float(f"{integer_part}.{raw[decimal_index + 1:]}")
    return _parse_float(re.sub(r"[.,]", "", raw))


def _parse_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).replace("$", "").replace("₮", "").strip()
    if "," in text and "." not in text and re.search(r",[0-9]{1,2}$", text):
        text = text.replace(",", ".")
    else:
        text = text.replace(",", "")
    try:
        number = float(text)
    except ValueError:
        return None
    return number if number >= 0 else None


def _parse_int(value: Any) -> int | None:
    if value is None:
        return None
    match = re.search(r"([0-9][0-9,]*)", str(value))
    if match:
        value = match.group(1)
    try:
        number = int(float(str(value).replace(",", "").strip()))
    except ValueError:
        return None
    return number if number >= 0 else None


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]
