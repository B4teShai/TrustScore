"""Optional market-reference enrichment from Serper shopping search."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import re
import threading
import time
from statistics import median
from typing import Any
from urllib.parse import urlparse, urlunparse

import httpx

from app.core.config import settings
from app.schemas.product_analysis import ProductPageData
from app.services.market_context import (
    expected_currency_for_market,
    market_country_from_url,
    market_from_url,
    normalize_currency_code,
    resolve_target_market,
    should_use_price_currency,
)


logger = logging.getLogger(__name__)

_PRICE_PREFIX_RE = re.compile(
    r"(US\s*\$|\$|USD|€|EUR|£|GBP|¥|￥|JPY)\s*([0-9][0-9,]*(?:\.[0-9]+)?)",
    re.I,
)
_PRICE_SUFFIX_RE = re.compile(
    r"([0-9][0-9,]*(?:\.[0-9]+)?)\s*(USD|EUR|GBP|JPY|円)",
    re.I,
)
_TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)
_SERPER_COUNT_SIGNAL_RE = re.compile(r"^market_reference:serper:count=(\d+)$")
_STOPWORDS = {
    "a",
    "and",
    "for",
    "in",
    "of",
    "on",
    "set",
    "the",
    "to",
    "with",
}
_MARKET_BY_CURRENCY = {
    "USD": "US",
    "JPY": "JP",
    "EUR": "EU",
    "GBP": "UK",
}
_SUPPORTED_REFERENCE_CURRENCIES = set(_MARKET_BY_CURRENCY)


@dataclass(frozen=True)
class MarketReference:
    """Median market reference from comparable listings."""

    median_price: float
    currency: str
    comparable_count: int
    source: str = "Serper"
    original_currency: str | None = None
    exchange_rate: float | None = None
    exchange_rate_source: str | None = None
    exchange_rate_date: str | None = None


@dataclass(frozen=True)
class _ListingPrice:
    price: float
    currency: str
    title: str
    link: str | None = None


@dataclass(frozen=True)
class ExchangeRate:
    """One unit of base currency converted into quote currency."""

    base: str
    quote: str
    rate: float
    source: str = "Frankfurter"
    date: str | None = None


@dataclass
class _CacheEntry:
    expires_at: float
    value: object


class FrankfurterExchangeRateProvider:
    """Fetch FX rates for converting market references to the listed currency."""

    def __init__(
        self,
        *,
        api_url: str,
        enabled: bool = True,
        timeout_seconds: float = 3.0,
        cache_ttl_seconds: float = 21600.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.enabled = enabled
        self.timeout_seconds = timeout_seconds
        self.cache_ttl_seconds = max(0.0, cache_ttl_seconds)
        self._http_client = http_client
        self._cache: dict[str, _CacheEntry] = {}
        self._lock = threading.Lock()

    def lookup(self, base: str, quote: str) -> ExchangeRate | None:
        base_currency = normalize_currency_code(base)
        quote_currency = normalize_currency_code(quote)
        if (
            not self.enabled
            or base_currency is None
            or quote_currency is None
            or base_currency not in _SUPPORTED_REFERENCE_CURRENCIES
            or quote_currency not in _SUPPORTED_REFERENCE_CURRENCIES
        ):
            return None
        if base_currency == quote_currency:
            return ExchangeRate(base=base_currency, quote=quote_currency, rate=1.0)

        cache_key = f"{base_currency}:{quote_currency}"
        cached = self._get_cache(cache_key)
        if cached is not _MISS:
            return cached if isinstance(cached, ExchangeRate) else None

        rate = self._lookup_uncached(base_currency, quote_currency)
        self._set_cache(cache_key, rate)
        return rate

    def _lookup_uncached(self, base: str, quote: str) -> ExchangeRate | None:
        url = f"{self.api_url}/rate/{base}/{quote}"
        try:
            if self._http_client is not None:
                response = self._http_client.get(url)
            else:
                response = httpx.get(url, timeout=self.timeout_seconds)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            logger.info(
                "exchange_rate_unavailable",
                extra={"error_type": type(exc).__name__, "base": base, "quote": quote},
            )
            return None

        if not isinstance(data, dict):
            return None
        raw_rate = data.get("rate")
        try:
            rate = float(raw_rate)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(rate) or rate <= 0:
            return None
        return ExchangeRate(
            base=base,
            quote=quote,
            rate=rate,
            date=_clean_text(data.get("date")),
        )

    def _get_cache(self, key: str) -> object:
        if self.cache_ttl_seconds <= 0:
            return _MISS
        now = time.monotonic()
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return _MISS
            if entry.expires_at <= now:
                self._cache.pop(key, None)
                return _MISS
            return entry.value

    def _set_cache(self, key: str, value: ExchangeRate | None) -> None:
        if self.cache_ttl_seconds <= 0:
            return
        with self._lock:
            self._cache[key] = _CacheEntry(
                expires_at=time.monotonic() + self.cache_ttl_seconds,
                value=value,
            )


class SerperMarketReferenceProvider:
    """Fetch comparable listing prices with the Serper Google Shopping API."""

    def __init__(
        self,
        *,
        api_key: str | None,
        api_url: str,
        enabled: bool = True,
        timeout_seconds: float = 4.0,
        cache_ttl_seconds: float = 21600.0,
        max_results: int = 12,
        min_results: int = 3,
        http_client: httpx.Client | None = None,
        exchange_rate_provider: FrankfurterExchangeRateProvider | None = None,
    ) -> None:
        self.api_key = api_key
        self.api_url = api_url
        self.enabled = enabled
        self.timeout_seconds = timeout_seconds
        self.cache_ttl_seconds = max(0.0, cache_ttl_seconds)
        self.max_results = max(1, max_results)
        self.min_results = max(1, min_results)
        self._http_client = http_client
        self._exchange_rate_provider = exchange_rate_provider
        self._cache: dict[str, _CacheEntry] = {}
        self._lock = threading.Lock()

    @property
    def active(self) -> bool:
        return bool(self.enabled and self.api_key)

    def lookup(
        self,
        product: ProductPageData,
        *,
        target_market: str | None = "auto",
        locale: str | None = None,
        allow_without_listed_price: bool = False,
    ) -> MarketReference | None:
        """Return a same-currency market reference, or None when not verified."""
        if (
            not self.active
            or product.average_market_price is not None
            or (product.price is None and not allow_without_listed_price)
            or not product.product_title.strip()
        ):
            return None

        resolved_market = resolve_target_market(target_market, url=product.url, locale=locale)
        expected_currency = expected_currency_for_market(resolved_market)
        currency = normalize_currency_code(product.currency) or expected_currency
        if not should_use_price_currency(currency, expected_currency=expected_currency, url=product.url):
            return None
        market = market_country_from_url(product.url) or market_from_url(product.url) or resolved_market

        query = _query_for_product(product.product_title)
        if not query:
            return None

        cache_key = f"{market}:{currency}:{query.lower()}"
        cached = self._get_cache(cache_key)
        if cached is not _MISS:
            return cached

        reference = self._lookup_uncached(
            query=query,
            product=product,
            currency=currency,
            market=market,
        )
        self._set_cache(cache_key, reference)
        return reference

    def _lookup_uncached(
        self,
        *,
        query: str,
        product: ProductPageData,
        currency: str,
        market: str,
    ) -> MarketReference | None:
        headers = {
            "Content-Type": "application/json",
            "X-API-KEY": self.api_key or "",
        }
        body: dict[str, object] = {
            "q": query,
            "gl": _serper_gl(market),
            "hl": _serper_hl(market),
            "num": self.max_results,
        }
        try:
            if self._http_client is not None:
                response = self._http_client.post(self.api_url, headers=headers, json=body)
            else:
                response = httpx.post(
                    self.api_url,
                    headers=headers,
                    json=body,
                    timeout=self.timeout_seconds,
                )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            logger.info(
                "serper_market_reference_unavailable",
                extra={"error_type": type(exc).__name__},
            )
            return None

        listings = _listing_prices_from_serper(
            data,
            product_title=product.product_title,
            product_url=product.url,
        )[: self.max_results]
        converted = self._convert_listing_prices(listings, target_currency=currency)
        if len(converted) < self.min_results:
            return None

        prices = [item[0] for item in converted]
        median_price = _round_money(float(median(prices)), currency)
        if not math.isfinite(median_price) or median_price <= 0:
            return None

        converted_rates = [item[1] for item in converted if item[1] is not None]
        exchange_rate = converted_rates[0] if converted_rates else None
        original_currency = None
        if exchange_rate is not None:
            converted_currencies = sorted({rate.base for rate in converted_rates})
            original_currency = converted_currencies[0] if len(converted_currencies) == 1 else "mixed"

        return MarketReference(
            median_price=median_price,
            currency=currency,
            comparable_count=len(converted),
            source="Serper",
            original_currency=original_currency,
            exchange_rate=exchange_rate.rate if exchange_rate else None,
            exchange_rate_source=exchange_rate.source if exchange_rate else None,
            exchange_rate_date=exchange_rate.date if exchange_rate else None,
        )

    def _convert_listing_prices(
        self,
        listings: list[_ListingPrice],
        *,
        target_currency: str,
    ) -> list[tuple[float, ExchangeRate | None]]:
        converted: list[tuple[float, ExchangeRate | None]] = []
        for listing in listings:
            if listing.currency == target_currency:
                converted.append((_round_money(listing.price, target_currency), None))
                continue
            if self._exchange_rate_provider is None:
                continue
            rate = self._exchange_rate_provider.lookup(listing.currency, target_currency)
            if rate is None:
                continue
            converted.append((_round_money(listing.price * rate.rate, target_currency), rate))
        return converted

    def _get_cache(self, key: str) -> MarketReference | None | object:
        if self.cache_ttl_seconds <= 0:
            return _MISS
        now = time.monotonic()
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return _MISS
            if entry.expires_at <= now:
                self._cache.pop(key, None)
                return _MISS
            return entry.value

    def _set_cache(self, key: str, value: MarketReference | None) -> None:
        if self.cache_ttl_seconds <= 0:
            return
        with self._lock:
            self._cache[key] = _CacheEntry(
                expires_at=time.monotonic() + self.cache_ttl_seconds,
                value=value,
            )


_MISS = object()


def enrich_market_reference(
    product: ProductPageData,
    *,
    target_market: str | None = "auto",
    locale: str | None = None,
    allow_without_listed_price: bool = False,
) -> tuple[ProductPageData, list[str]]:
    """Add a Serper median market price when a verified reference is available."""
    if not market_reference_provider.active:
        return product, ["market_reference:unavailable:provider_inactive"]
    if product.average_market_price is not None:
        return product, []
    if product.price is None and not allow_without_listed_price:
        return product, ["market_reference:unavailable:missing_listed_price"]
    resolved_market = resolve_target_market(target_market, url=product.url, locale=locale)
    expected_currency = expected_currency_for_market(resolved_market)
    currency = normalize_currency_code(product.currency) or expected_currency
    if not should_use_price_currency(currency, expected_currency=expected_currency, url=product.url):
        return product, [f"market_reference:unavailable:currency_mismatch:{currency}"]

    reference = market_reference_provider.lookup(
        product,
        target_market=target_market,
        locale=locale,
        allow_without_listed_price=allow_without_listed_price,
    )
    if reference is None:
        return product, ["market_reference:unavailable:no_verified_comparables"]
    enriched = product.model_copy(
        update={
            "average_market_price": reference.median_price,
            "currency": reference.currency,
            "market_reference_count": reference.comparable_count,
            "market_reference_source": reference.source,
            "market_reference_original_currency": reference.original_currency,
            "market_reference_exchange_rate": reference.exchange_rate,
            "market_reference_exchange_rate_source": reference.exchange_rate_source,
            "market_reference_exchange_rate_date": reference.exchange_rate_date,
        }
    )
    return enriched, [f"market_reference:serper:count={reference.comparable_count}"]


def market_reference_count_from_signals(signals: list[str]) -> int | None:
    for signal in signals:
        match = _SERPER_COUNT_SIGNAL_RE.match(signal)
        if match:
            return int(match.group(1))
    return None


def _listing_prices_from_serper(
    data: object,
    *,
    product_title: str,
    product_url: str,
) -> list[_ListingPrice]:
    if not isinstance(data, dict):
        return []
    raw_items = data.get("shopping")
    if not isinstance(raw_items, list):
        return []

    title_tokens = _tokens(product_title)
    out: list[_ListingPrice] = []
    seen: set[tuple[str, float]] = set()
    product_url_key = _canonical_url(product_url)
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        title = _clean_text(item.get("title"))
        if not title or not _similar_enough(title_tokens, title):
            continue
        link = _clean_text(item.get("link"))
        if link and _canonical_url(link) == product_url_key:
            continue
        price_text = _clean_text(item.get("price")) or _clean_text(item.get("snippet"))
        parsed = _parse_price(price_text)
        if parsed is None:
            continue
        key = ((link or title).lower(), parsed.price)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            _ListingPrice(
                price=parsed.price,
                currency=parsed.currency,
                title=title,
                link=link,
            )
        )
    return out


def _parse_price(value: str | None) -> _ListingPrice | None:
    if not value:
        return None
    for regex in (_PRICE_PREFIX_RE, _PRICE_SUFFIX_RE):
        match = regex.search(value)
        if not match:
            continue
        if regex is _PRICE_PREFIX_RE:
            raw_currency, raw_price = match.group(1), match.group(2)
        else:
            raw_price, raw_currency = match.group(1), match.group(2)
        currency = normalize_currency_code(raw_currency)
        if currency not in _SUPPORTED_REFERENCE_CURRENCIES:
            return None
        price = _number(raw_price)
        if price is None:
            return None
        return _ListingPrice(price=price, currency=currency, title="")
    return None


def _query_for_product(title: str) -> str:
    words = title.split()
    query = " ".join(words[:14])
    return query[:180].strip()


def _tokens(value: str) -> set[str]:
    return {
        token.lower()
        for token in _TOKEN_RE.findall(value)
        if len(token) > 2 and token.lower() not in _STOPWORDS
    }


def _similar_enough(product_tokens: set[str], listing_title: str) -> bool:
    if not product_tokens:
        return False
    listing_tokens = _tokens(listing_title)
    if not listing_tokens:
        return False
    overlap = len(product_tokens & listing_tokens)
    required = 1 if len(product_tokens) <= 3 else max(2, math.ceil(len(product_tokens) * 0.28))
    return overlap >= required


def _number(value: str) -> float | None:
    try:
        cleaned = value.replace(",", "")
        parsed = float(cleaned)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) and parsed > 0 else None


def _round_money(value: float, currency: str) -> float:
    if currency.upper() in {"JPY", "KRW", "VND", "CLP", "ISK"}:
        return float(round(value))
    return round(value, 2)


def _clean_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = " ".join(value.split())
    return cleaned or None


def _canonical_url(value: str) -> str:
    try:
        parsed = urlparse(value)
    except ValueError:
        return value.strip().lower()
    return urlunparse(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path.rstrip("/"),
            "",
            "",
            "",
        )
    )


def _serper_gl(market: str) -> str:
    if market == "JP":
        return "jp"
    if market == "UK":
        return "gb"
    if market in {"FR", "DE", "IT", "ES", "NL", "BE", "IE", "AT", "PT", "PL", "SE"}:
        return market.lower()
    if market == "EU":
        return "de"
    return "us"


def _serper_hl(market: str) -> str:
    if market == "JP":
        return "ja"
    if market == "FR":
        return "fr"
    if market == "DE":
        return "de"
    if market in {"IT", "ES", "NL", "PT", "PL", "SE"}:
        return market.lower()
    if market == "UK":
        return "en"
    if market == "EU":
        return "en"
    return "en"


exchange_rate_provider = FrankfurterExchangeRateProvider(
    api_url=settings.exchange_rate_api_url,
    enabled=settings.exchange_rate_enabled,
    timeout_seconds=settings.exchange_rate_timeout_seconds,
    cache_ttl_seconds=settings.exchange_rate_cache_ttl_seconds,
)


market_reference_provider = SerperMarketReferenceProvider(
    api_key=settings.serper_api_key,
    api_url=settings.serper_api_url,
    enabled=settings.market_reference_enabled,
    timeout_seconds=settings.market_reference_timeout_seconds,
    cache_ttl_seconds=settings.market_reference_cache_ttl_seconds,
    max_results=settings.market_reference_max_results,
    min_results=settings.market_reference_min_results,
    exchange_rate_provider=exchange_rate_provider,
)
