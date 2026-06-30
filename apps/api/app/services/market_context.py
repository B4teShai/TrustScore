"""Target-market and currency policy for product scans."""

from __future__ import annotations

from typing import Literal
from urllib.parse import urlparse


TargetMarket = Literal["auto", "US", "JP", "EU", "UK"]
ResolvedTargetMarket = Literal["US", "JP", "EU", "UK"]

SUPPORTED_TARGET_MARKETS = {"auto", "US", "JP", "EU", "UK"}
TARGET_MARKET_CURRENCIES: dict[ResolvedTargetMarket, str] = {
    "US": "USD",
    "JP": "JPY",
    "EU": "EUR",
    "UK": "GBP",
}
SUPPORTED_PRICE_CURRENCIES = set(TARGET_MARKET_CURRENCIES.values())
EU_HOST_MARKERS = (
    ".de",
    ".fr",
    ".it",
    ".es",
    ".nl",
    ".be",
    ".ie",
    ".at",
    ".pt",
    ".pl",
    ".se",
)
EU_LANGUAGE_CODES = {"de", "fr", "it", "es", "nl", "pt", "pl", "sv"}


def normalize_target_market(value: str | None) -> TargetMarket:
    """Return a supported target market token, defaulting to auto."""
    normalized = (value or "auto").strip().upper()
    if normalized == "AUTO":
        return "auto"
    if normalized in {"US", "JP", "EU", "UK"}:
        return normalized  # type: ignore[return-value]
    return "auto"


def resolve_target_market(
    value: str | None,
    *,
    url: str,
    locale: str | None = None,
) -> ResolvedTargetMarket:
    """Resolve auto market from URL first, then locale, then US."""
    requested = normalize_target_market(value)
    if requested != "auto":
        return requested

    host_market = market_from_url(url)
    if host_market is not None:
        return host_market

    lang = (locale or "").strip().lower().replace("_", "-")
    primary = lang.split("-", 1)[0]
    if primary == "ja" or lang.endswith("-jp"):
        return "JP"
    if lang in {"en-gb", "en-uk"} or lang.endswith("-gb") or lang.endswith("-uk"):
        return "UK"
    if primary in EU_LANGUAGE_CODES:
        return "EU"
    return "US"


def market_from_url(url: str) -> ResolvedTargetMarket | None:
    """Infer a market from marketplace/domain geography without using page locale."""
    host = (urlparse(url).hostname or url).lower().strip(".")
    if host.endswith(".co.jp") or host.endswith(".jp"):
        return "JP"
    if host.endswith(".co.uk") or host.endswith(".uk"):
        return "UK"
    if any(host.endswith(marker) for marker in EU_HOST_MARKERS):
        return "EU"
    if (
        host == "amazon.com"
        or host.endswith(".amazon.com")
        or host == "ebay.com"
        or host.endswith(".ebay.com")
        or host == "etsy.com"
        or host.endswith(".etsy.com")
    ):
        return "US"
    return None


def expected_currency_for_market(market: ResolvedTargetMarket) -> str:
    return TARGET_MARKET_CURRENCIES[market]


def normalize_currency_code(value: str | None) -> str | None:
    if not value:
        return None
    token = value.strip().upper().replace(" ", "")
    text = value.strip().upper()
    if token in {"US$", "$", "USD"}:
        return "USD"
    if token in {"¥", "￥", "円", "JPY", "JP¥"}:
        return "JPY"
    if token in {"€", "EUR"}:
        return "EUR"
    if token in {"£", "GBP"}:
        return "GBP"
    if token in {"MNT", "₮"}:
        return "MNT"
    for code in ("USD", "JPY", "EUR", "GBP", "MNT"):
        if code in text.replace("_", " ").replace("-", " ").split():
            return code
    if len(token) == 3 and token.isalpha():
        return token
    return token[:16] or None


def is_marketplace_host(url_or_host: str | None) -> bool:
    if not url_or_host:
        return False
    host = urlparse(url_or_host).hostname or url_or_host
    host = host.lower().strip(".")
    return (
        "amazon." in host
        or host.endswith("ebay.com")
        or ".ebay." in host
        or host.endswith("etsy.com")
        or ".etsy." in host
    )


def should_use_price_currency(
    currency: str | None,
    *,
    expected_currency: str,
    url: str,
) -> bool:
    """Reject unsupported currencies; market references still require same-currency listings."""
    normalized = normalize_currency_code(currency)
    if normalized is None:
        return True
    return normalized in SUPPORTED_PRICE_CURRENCIES
