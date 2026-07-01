"""Bounded retrieval for public product pages."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import ipaddress
import socket
import time
from typing import Iterator
from urllib.parse import urljoin, urlparse, urlunparse

import httpx


MAX_RESPONSE_BYTES = 2_000_000
MAX_REDIRECTS = 4
REQUEST_TIMEOUT_SECONDS = 8.0
TRANSIENT_RETRY_STATUS_CODES = {429, 500, 502, 503, 504}
FETCH_RETRY_BACKOFF_SECONDS = (0.5, 1.5)
RENDERED_PRODUCT_SELECTORS = (
    # Amazon product detail surface
    "#productTitle",
    "[data-feature-name='title']",
    "#landingImage",
    "#imgTagWrapperId",
    "#corePrice_feature_div",
    "#price",
    ".a-price",
    "#bylineInfo",
    "#sellerProfileTriggerId",
    "#merchant-info",
    "[tabular-attribute-name='Sold by']",
    "#acrPopover",
    "#acrCustomerReviewText",
    "#feature-bullets",
    "#productDescription",
    "#availability",
    "#detailBullets_feature_div",
    "#prodDetails",
    "#reviewsMedley",
    "[data-hook='review']",
    "[data-hook='reviewText']",
    "[id^='social-proofing']",
    # eBay product detail surface
    "h1.x-item-title__mainTitle",
    "[data-testid='x-item-title']",
    "img#icImg",
    "[data-testid='ux-image-carousel-item']",
    ".x-price-primary",
    ".x-price-approx",
    "[data-testid='x-price-primary']",
    "[data-testid='x-sellercard-atf__info']",
    ".x-sellercard-atf__info__about-seller",
    "[data-testid='x-sellercard-atf__data-item']",
    "[data-testid='ux-seller-section__item']",
    "[data-testid='x-returns-minview']",
    "[data-testid='ux-labels-values-Returns']",
    "[data-review-region]",
    "[data-review-text]",
    # Etsy listing detail surface
    "h1[data-buy-box-listing-title]",
    "[data-buy-box-region='title']",
    "[data-carousel]",
    ".listing-page-image-carousel-component",
    "[data-listing-page-image]",
    "[data-buy-box-region='price']",
    "[data-buy-box-region='shop-name']",
    "[data-region='shop-name']",
    "[data-region='shop-rating']",
    "[data-region='listing-page-description']",
    "[data-id='description-text']",
    "[data-region='listing-page-policies']",
    "[data-region='return-policy']",
    "h1",
    "button",
)
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "TrustScoreProductAnalyzer/0.1"
)


class PageFetchError(RuntimeError):
    """Raised when a public product page cannot be fetched."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class URLValidationError(ValueError):
    """Raised when a URL is not safe for the product-page fetcher."""


@dataclass(frozen=True)
class FetchedPage:
    """HTML content returned by either static or rendered page retrieval."""

    requested_url: str
    final_url: str
    html: str
    mode: str


def validate_public_web_url(raw_url: str) -> str:
    """Return a normalized public HTTP(S) URL or raise a validation error."""
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
    if port is not None and port != _default_port(parsed.scheme.lower()):
        raise URLValidationError("Only default HTTP and HTTPS ports are supported.")

    hostname = parsed.hostname.strip().lower().rstrip(".")
    if hostname in {"localhost", "localhost.localdomain"} or hostname.endswith(".localhost"):
        raise URLValidationError("Local machine URLs are not supported.")

    _validate_public_hostname(hostname)
    normalized = parsed._replace(
        scheme=parsed.scheme.lower(),
        netloc=_normalized_netloc(hostname, port),
        fragment="",
    )
    return urlunparse(normalized)


def fetch_public_html(url: str) -> FetchedPage:
    """Fetch a public web page, retrying transient failures with backoff."""
    last_error: PageFetchError | None = None
    for attempt, backoff_seconds in enumerate((*FETCH_RETRY_BACKOFF_SECONDS, None)):
        try:
            return _fetch_public_html_once(url)
        except PageFetchError as exc:
            last_error = exc
            if exc.status_code not in TRANSIENT_RETRY_STATUS_CODES or backoff_seconds is None:
                raise
            time.sleep(backoff_seconds)
    raise last_error  # pragma: no cover - loop always returns or raises


def _fetch_public_html_once(url: str) -> FetchedPage:
    current_url = validate_public_web_url(url)
    headers = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.5",
        "User-Agent": USER_AGENT,
    }

    try:
        with httpx.Client(
            timeout=REQUEST_TIMEOUT_SECONDS,
            headers=headers,
            trust_env=False,
        ) as client:
            for _redirect_index in range(MAX_REDIRECTS + 1):
                with (
                    _pinned_dns_resolution(current_url),
                    client.stream("GET", current_url, follow_redirects=False) as response,
                ):
                    if 300 <= response.status_code < 400 and response.headers.get("location"):
                        current_url = validate_public_web_url(
                            urljoin(current_url, response.headers["location"])
                        )
                        continue

                    if response.status_code >= 400:
                        raise PageFetchError(
                            f"Product page returned HTTP status {response.status_code}.",
                            status_code=response.status_code,
                        )

                    validate_public_web_url(str(response.url))
                    _validate_html_content_type(response.headers.get("content-type"))
                    content_length = _parse_content_length(response.headers.get("content-length"))
                    if content_length and content_length > MAX_RESPONSE_BYTES:
                        raise PageFetchError("Product page response is too large.")

                    chunks: list[bytes] = []
                    total = 0
                    for chunk in response.iter_bytes():
                        total += len(chunk)
                        if total > MAX_RESPONSE_BYTES:
                            raise PageFetchError("Product page response is too large.")
                        chunks.append(chunk)

                    encoding = response.encoding or "utf-8"
                    html = b"".join(chunks).decode(encoding, errors="replace")
                    return FetchedPage(
                        requested_url=url,
                        final_url=str(response.url),
                        html=html,
                        mode="http",
                    )
    except httpx.RequestError as exc:
        raise PageFetchError("Product page request failed.") from exc

    raise PageFetchError("Product page redirected too many times.")


def render_public_html(url: str) -> FetchedPage:
    """Render a page with Playwright when that optional runtime is installed."""
    safe_url = validate_public_web_url(url)
    parsed_safe_url = urlparse(safe_url)
    host = parsed_safe_url.hostname or ""
    host_ip = _resolve_public_addresses(host)[0]
    resolver_rules = f"MAP {host} {host_ip}"
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:  # pragma: no cover - depends on optional runtime
        raise PageFetchError("Rendered page fallback is not installed.") from exc

    browser = None
    try:  # pragma: no cover - exercised with a mocked fallback in tests
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                headless=True,
                args=[f"--host-resolver-rules={resolver_rules}"],
            )
            page = browser.new_page(user_agent=USER_AGENT)
            page.route("**/*", lambda route: _safe_playwright_route(route, host))
            page.goto(
                safe_url,
                wait_until="domcontentloaded",
                timeout=REQUEST_TIMEOUT_SECONDS * 1000,
            )
            html = _compact_rendered_html(page)
            final_url = page.url
            validate_public_web_url(final_url)
            if len(html.encode("utf-8")) > MAX_RESPONSE_BYTES:
                raise PageFetchError("Rendered product snapshot is too large.")
            return FetchedPage(
                requested_url=url,
                final_url=final_url,
                html=html,
                mode="rendered",
            )
    except PageFetchError:
        raise
    except Exception as exc:
        raise PageFetchError("Rendered page fallback failed.") from exc
    finally:
        if browser is not None:
            try:
                browser.close()
            except Exception:
                pass


def _normalized_netloc(hostname: str, port: int | None) -> str:
    if port is None:
        return hostname
    return f"{hostname}:{port}"


def _default_port(scheme: str) -> int:
    return 80 if scheme == "http" else 443


def _validate_public_hostname(hostname: str) -> None:
    _resolve_public_addresses(hostname)


def _resolve_public_addresses(hostname: str) -> list[str]:
    try:
        parsed_ip = ipaddress.ip_address(hostname)
        _validate_public_ip(parsed_ip)
        return [str(parsed_ip)]
    except ValueError:
        pass

    try:
        infos = socket.getaddrinfo(hostname, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise URLValidationError("URL hostname could not be resolved.") from exc

    addresses = {info[4][0] for info in infos}
    if not addresses:
        raise URLValidationError("URL hostname could not be resolved.")

    public_addresses = sorted(addresses)
    for address in public_addresses:
        _validate_public_ip(ipaddress.ip_address(address))
    return public_addresses


def _validate_public_ip(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> None:
    if (
        address.is_loopback
        or address.is_private
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    ):
        raise URLValidationError("Only public web URLs are supported.")


def _parse_content_length(value: str | None) -> int | None:
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _validate_html_content_type(content_type: str | None) -> None:
    if not content_type:
        return
    media_type = content_type.split(";", 1)[0].strip().lower()
    if media_type not in {
        "text/html",
        "application/xhtml+xml",
        "application/xml",
        "text/xml",
    }:
        raise PageFetchError("Product page response is not HTML.")


@contextmanager
def _pinned_dns_resolution(url: str) -> Iterator[None]:
    parsed = urlparse(url)
    hostname = parsed.hostname
    if not hostname:
        raise URLValidationError("URL must include a hostname.")

    pinned_addresses = _resolve_public_addresses(hostname)
    original_getaddrinfo = socket.getaddrinfo

    def getaddrinfo(
        host,
        port,
        family=0,
        type=0,
        proto=0,
        flags=0,
    ):
        host_text = host.decode("ascii", errors="ignore") if isinstance(host, bytes) else str(host)
        if host_text.lower().rstrip(".") != hostname.lower().rstrip("."):
            return original_getaddrinfo(host, port, family, type, proto, flags)

        results = []
        for address_text in pinned_addresses:
            address = ipaddress.ip_address(address_text)
            address_family = socket.AF_INET6 if address.version == 6 else socket.AF_INET
            if family not in (0, address_family):
                continue
            socket_type = type or socket.SOCK_STREAM
            protocol = proto or socket.IPPROTO_TCP
            sockaddr = (
                (address_text, port, 0, 0)
                if address_family == socket.AF_INET6
                else (address_text, port)
            )
            results.append((address_family, socket_type, protocol, "", sockaddr))

        if not results:
            raise socket.gaierror(socket.EAI_NONAME, "No pinned address for requested family.")
        return results

    socket.getaddrinfo = getaddrinfo
    try:
        yield
    finally:
        socket.getaddrinfo = original_getaddrinfo


def _compact_rendered_html(page) -> str:
    return page.evaluate(
        """
        (selectors) => {
          const chunks = [];
          const push = (value) => {
            if (value && typeof value === "string") {
              chunks.push(value);
            }
          };

          push(`<title>${document.title || ""}</title>`);

          document
            .querySelectorAll("meta[property], meta[name]")
            .forEach((node) => {
              const key = node.getAttribute("property") || node.getAttribute("name") || "";
              if (/^(og:|product:|twitter:|description|brand)/i.test(key)) {
                push(node.outerHTML);
              }
            });

          document
            .querySelectorAll('script[type="application/ld+json"]')
            .forEach((node, index) => {
              if (index < 6) {
                push(node.outerHTML);
              }
            });

          selectors.forEach((selector) => {
            document.querySelectorAll(selector).forEach((node, index) => {
              if (index < 8) {
                push(node.outerHTML);
              }
            });
          });

          return `<!doctype html><html><head></head><body>${chunks.join("\\n")}</body></html>`;
        }
        """,
        list(RENDERED_PRODUCT_SELECTORS),
    )


def _safe_playwright_route(route, allowed_hostname: str) -> None:
    try:  # pragma: no cover - depends on optional Playwright runtime
        safe_url = validate_public_web_url(route.request.url)
        request_hostname = (urlparse(safe_url).hostname or "").lower().rstrip(".")
        if request_hostname != allowed_hostname.lower().rstrip("."):
            _abort_route(route)
            return
    except URLValidationError:
        _abort_route(route)
        return
    _continue_route(route)


def _abort_route(route) -> None:
    try:  # pragma: no cover - depends on optional Playwright runtime
        route.abort()
    except BaseException:
        pass


def _continue_route(route) -> None:
    try:  # pragma: no cover - depends on optional Playwright runtime
        route.continue_()
    except BaseException:
        pass
