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


PRICE_RE = re.compile(r"(\$|USD|MNT|₮|€|£)\s?([0-9][0-9,.]*)", re.I)
RATING_RE = re.compile(r"([0-5](?:\.[0-9])?)\s*(?:out of 5|stars?)", re.I)
REVIEW_COUNT_RE = re.compile(r"([0-9][0-9,]*)\s+(?:ratings?|reviews?)", re.I)
POLICY_RE = re.compile(r"(.{0,80}(?:return|refund|exchange|warranty).{0,160})", re.I)
PRODUCT_URL_RE = re.compile(r"/(dp|gp/product|product|products|item|itm)/", re.I)
BOUGHT_RECENT_RE = re.compile(
    r"([0-9][0-9.,]*)\s*([KkMm])?\+?\s+bought\s+in\s+past\s+month", re.I
)


@dataclass(frozen=True)
class ExtractionResult:
    """Product extraction result plus detection metadata."""

    detected: bool
    product: ProductPageData | None
    reason: str
    signals: list[str] = field(default_factory=list)


def extract_product_page(html: str, url: str) -> ExtractionResult:
    """Extract one product-page payload from HTML."""
    soup = BeautifulSoup(html, "html.parser")
    page_text = soup.get_text(" ", strip=True)
    product_objects = _json_ld_products(soup)
    title = _clean_title(_first_product_value(product_objects, "name") or _find_title(soup))
    site = urlparse(url).hostname

    if not title:
        return ExtractionResult(
            detected=False,
            product=None,
            reason="No clear product title was found.",
        )

    price, currency = _extract_price_and_currency(soup, page_text, product_objects)
    seller_name = _extract_seller(soup, page_text, product_objects)
    reviews = _extract_reviews(soup, product_objects)
    product = ProductPageData(
        url=url,
        site=site,
        product_title=title,
        description=_extract_description(soup, product_objects),
        product_image_url=_extract_image(soup, url, product_objects),
        price=price,
        currency=currency,
        average_market_price=None,
        seller=SellerInfo(name=seller_name) if seller_name else None,
        return_policy=_extract_policy(page_text),
        reviews=reviews,
        rating=_extract_rating(soup, page_text, product_objects),
        review_count=_extract_review_count(soup, page_text, product_objects),
        units_bought_recent=_extract_units_bought_recent(page_text),
    )
    signals = _product_signals(product, page_text, bool(product_objects), url)
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


def _find_title(soup: BeautifulSoup) -> str | None:
    selectors = [
        "#productTitle",
        "[data-testid='product-title']",
        "[itemprop='name']",
        "h1",
        "meta[property='og:title']",
        "meta[name='twitter:title']",
        "title",
    ]
    for selector in selectors:
        value = _element_text(soup.select_one(selector))
        if value:
            return value
    return None


def _clean_title(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = re.sub(r"\s+", " ", value).strip()
    cleaned = re.sub(r"\s+[-|]\s+Amazon(?:\.com)?$", "", cleaned, flags=re.I)
    if len(cleaned) < 4:
        return None
    return cleaned[:180]


def _extract_description(soup: BeautifulSoup, products: list[dict[str, Any]]) -> str | None:
    value = _first_product_value(products, "description")
    if value:
        return _compact(value)[:500]
    return _compact(_element_text(soup.select_one("meta[name='description']")) or "")[:500] or None


def _extract_image(
    soup: BeautifulSoup,
    page_url: str,
    products: list[dict[str, Any]],
) -> str | None:
    for product in products:
        image_url = _image_from_unknown(product.get("image"), page_url)
        if image_url:
            return image_url

    selectors = [
        "meta[property='og:image']",
        "meta[name='twitter:image']",
        "#landingImage",
        "[itemprop='image']",
        "img",
    ]
    for selector in selectors:
        element = soup.select_one(selector)
        image_url = _image_from_element(element, page_url)
        if image_url:
            return image_url
    return None


def _extract_price_and_currency(
    soup: BeautifulSoup,
    page_text: str,
    products: list[dict[str, Any]],
) -> tuple[float | None, str | None]:
    for product in products:
        for offer in _as_list(product.get("offers")):
            if not isinstance(offer, dict):
                continue
            price = _parse_float(offer.get("price"))
            if price is not None:
                currency = _string_from_unknown(offer.get("priceCurrency"))
                return price, currency

    meta_price = _element_text(soup.select_one("meta[property='product:price:amount']"))
    parsed_meta_price = _parse_float(meta_price)
    if parsed_meta_price is not None:
        currency = _element_text(soup.select_one("meta[property='product:price:currency']"))
        return parsed_meta_price, currency

    match = PRICE_RE.search(page_text)
    if not match:
        return None, None
    price = _parse_float(match.group(2))
    currency = match.group(1).upper()
    return price, currency


def _extract_rating(
    soup: BeautifulSoup,
    page_text: str,
    products: list[dict[str, Any]],
) -> float | None:
    for product in products:
        aggregate = product.get("aggregateRating")
        if isinstance(aggregate, dict):
            rating = _parse_float(aggregate.get("ratingValue"))
            if rating is not None and 0 <= rating <= 5:
                return rating

    rating = _parse_float(_element_text(soup.select_one("[itemprop='ratingValue']")))
    if rating is not None and 0 <= rating <= 5:
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
) -> int | None:
    for product in products:
        aggregate = product.get("aggregateRating")
        if isinstance(aggregate, dict):
            count = _parse_int(aggregate.get("reviewCount") or aggregate.get("ratingCount"))
            if count is not None:
                return count

    count = _parse_int(_element_text(soup.select_one("[itemprop='reviewCount']")))
    if count is not None:
        return count

    match = REVIEW_COUNT_RE.search(page_text)
    return _parse_int(match.group(1)) if match else None


def _extract_seller(
    soup: BeautifulSoup,
    page_text: str,
    products: list[dict[str, Any]],
) -> str | None:
    for product in products:
        seller = _seller_from_unknown(product.get("seller") or product.get("brand"))
        if seller:
            return seller
        for offer in _as_list(product.get("offers")):
            seller = _seller_from_unknown(offer.get("seller") if isinstance(offer, dict) else None)
            if seller:
                return seller

    selectors = [
        "#sellerProfileTriggerId",
        "#merchant-info",
        "#bylineInfo",
        "[data-testid*='seller']",
        "[class*='seller']",
        "[id*='seller']",
        "[class*='merchant']",
        "[id*='merchant']",
    ]
    for selector in selectors:
        seller = _clean_seller(_element_text(soup.select_one(selector)))
        if seller:
            return seller
    return _clean_seller(page_text)


def _extract_policy(page_text: str) -> str | None:
    match = POLICY_RE.search(page_text)
    return _compact(match.group(1)) if match else None


def _extract_reviews(soup: BeautifulSoup, products: list[dict[str, Any]]) -> list[ReviewInput]:
    reviews: list[ReviewInput] = []
    seen: set[str] = set()
    for product in products:
        for raw_review in _as_list(product.get("review")):
            if not isinstance(raw_review, dict):
                continue
            text = _compact(
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

    # Current Amazon layout: each card is data-hook="review" with the body in
    # data-hook="reviewText". Prefer that precise body over the whole card text.
    for card in soup.select("[data-hook='review'], [data-hook='cmps-review']"):
        body = card.select_one("[data-hook='reviewText'], [data-hook='reviewTextContainer']")
        text = _compact((body or card).get_text(" ", strip=True))
        if len(text) < 20 or text in seen:
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
        text = _compact(element.get_text(" ", strip=True))
        if len(text) < 20 or text in seen:
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
) -> list[str]:
    signals: list[str] = []
    if has_structured_product:
        signals.append("structured_product")
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
    if re.search(r"\b(add to cart|add to bag|buy now|checkout|in stock)\b", page_text, re.I):
        signals.append("commerce_action")
    if PRODUCT_URL_RE.search(urlparse(url).path):
        signals.append("product_url")
    return signals


def _is_product_detail(signals: list[str]) -> bool:
    has = signals.__contains__
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


def _element_text(element: Any) -> str | None:
    if element is None:
        return None
    if getattr(element, "name", "") == "meta":
        return _compact(element.get("content") or "")
    return _compact(element.get_text(" ", strip=True))


def _compact(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


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
    cleaned = re.sub(r"[|•].*$", "", cleaned).strip()
    return cleaned if 2 <= len(cleaned) <= 80 else None


def _parse_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(str(value).replace("$", "").replace(",", "").strip())
    except ValueError:
        return None
    return number if number >= 0 else None


def _parse_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        number = int(float(str(value).replace(",", "").strip()))
    except ValueError:
        return None
    return number if number >= 0 else None


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]
