import pytest

from app.extraction.product_page import extract_product_page
from app.page_fetching import fetcher
from app.page_fetching.fetcher import (
    PageFetchError,
    URLValidationError,
    fetch_public_html,
    validate_public_web_url,
)
from app.services import product_page_analysis as page_service
from app.services.product_page_analysis import ProductPageAnalysis


PRODUCT_HTML = """
<html>
  <head>
    <meta property="og:image" content="https://example.com/img.jpg" />
    <script type="application/ld+json">
      {
        "@type": "Product",
        "name": "Portable Charger",
        "description": "Compact charger for travel",
        "image": "https://example.com/charger.jpg",
        "offers": {
          "price": "24.99",
          "priceCurrency": "USD",
          "seller": {"name": "Example Store"}
        },
        "aggregateRating": {"ratingValue": "4.4", "reviewCount": "118"},
        "review": [
          {"reviewBody": "Works well and shipped fast.", "reviewRating": {"ratingValue": "5"}}
        ]
      }
    </script>
  </head>
  <body>
    <button>Add to cart</button>
    <p>30-day return and refund policy available.</p>
  </body>
</html>
"""


def test_extract_product_page_from_json_ld() -> None:
    result = extract_product_page(PRODUCT_HTML, "https://example.com/product/charger")

    assert result.detected is True
    assert result.product is not None
    assert result.product.product_title == "Portable Charger"
    assert result.product.price == 24.99
    assert result.product.currency == "USD"
    assert result.product.seller is not None
    assert result.product.seller.name == "Example Store"
    assert result.product.reviews[0].text == "Works well and shipped fast."


def test_extract_product_page_rejects_weak_page() -> None:
    result = extract_product_page(
        "<html><title>About us</title><body>Welcome to our company.</body></html>",
        "https://example.com/about",
    )

    assert result.detected is False
    assert result.product is None


def test_extract_product_page_from_shopify_like_markup() -> None:
    html = """
    <html>
      <head>
        <meta property="og:title" content="Canvas Tote Bag" />
        <meta property="og:image" content="/images/tote.jpg" />
      </head>
      <body>
        <h1>Canvas Tote Bag</h1>
        <span class="price">$32.00</span>
        <button>Add to cart</button>
      </body>
    </html>
    """

    result = extract_product_page(html, "https://example.com/products/tote")

    assert result.detected is True
    assert result.product is not None
    assert result.product.product_title == "Canvas Tote Bag"
    assert result.product.product_image_url == "https://example.com/images/tote.jpg"


def test_extract_product_page_rejects_search_page_with_commerce_words() -> None:
    html = """
    <html>
      <title>Search results for chargers</title>
      <body>
        <h1>Search results</h1>
        <p>Add to cart buttons may appear on product grids.</p>
      </body>
    </html>
    """

    result = extract_product_page(html, "https://example.com/search?q=chargers")

    assert result.detected is False


def test_extract_product_page_uses_srcset_image() -> None:
    html = """
    <html>
      <body>
        <h1>Trail Camera</h1>
        <img srcset="/small.jpg 320w, /large.jpg 900w" />
        <span>$49.99</span>
        <button>Buy now</button>
      </body>
    </html>
    """

    result = extract_product_page(html, "https://example.com/product/camera")

    assert result.detected is True
    assert result.product is not None
    assert result.product.product_image_url == "https://example.com/large.jpg"


def test_extract_product_page_uses_dynamic_image_json() -> None:
    html = """
    <html>
      <body>
        <h1>Portable Speaker</h1>
        <img
          id="landingImage"
          data-a-dynamic-image='{
            "https://example.com/small.jpg": [100, 100],
            "https://example.com/large.jpg": [800, 800]
          }'
        />
        <span>$59.99</span>
        <button>Add to cart</button>
      </body>
    </html>
    """

    result = extract_product_page(html, "https://example.com/dp/speaker")

    assert result.detected is True
    assert result.product is not None
    assert result.product.product_image_url == "https://example.com/large.jpg"


@pytest.mark.parametrize(
    "image_markup",
    [
        '<meta property="og:image" content="http://127.0.0.1/admin.png" />',
        '<meta property="og:image" content="http://10.0.0.5/product.png" />',
        '<script type="application/ld+json">{"@type":"Product","name":"Desk Lamp","image":"https://shop.localhost/product.png","offers":{"price":"19.99","priceCurrency":"USD"}}</script>',
    ],
)
def test_extract_product_page_drops_private_image_urls(image_markup: str) -> None:
    html = f"""
    <html>
      <head>{image_markup}</head>
      <body>
        <h1>Desk Lamp</h1>
        <span>$19.99</span>
        <button>Add to cart</button>
      </body>
    </html>
    """

    result = extract_product_page(html, "https://example.com/products/lamp")

    assert result.detected is True
    assert result.product is not None
    assert result.product.product_image_url is None


@pytest.mark.parametrize(
    "url",
    [
        "file:///tmp/page.html",
        "http://localhost:8000/product",
        "http://127.0.0.1/product",
        "https://user:pass@example.com/product",
        "https://example.com:444/product",
        "http://example.com:99999/product",
    ],
)
def test_validate_public_web_url_rejects_non_public_targets(url: str) -> None:
    with pytest.raises(URLValidationError):
        validate_public_web_url(url)


def test_fetch_public_html_rejects_non_html_response(monkeypatch) -> None:
    monkeypatch.setattr(fetcher, "_resolve_public_addresses", lambda _hostname: ["93.184.216.34"])

    class Response:
        status_code = 200
        headers = {"content-type": "application/json"}
        encoding = "utf-8"
        url = "https://example.com/product"

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def iter_bytes(self):
            yield b'{"ok": true}'

    class Client:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def stream(self, *_args, **_kwargs):
            return Response()

    monkeypatch.setattr(fetcher.httpx, "Client", Client)

    with pytest.raises(PageFetchError, match="not HTML"):
        fetch_public_html("https://example.com/product")


def test_fetch_public_html_wraps_network_errors(monkeypatch) -> None:
    monkeypatch.setattr(fetcher, "_resolve_public_addresses", lambda _hostname: ["93.184.216.34"])

    class Client:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def stream(self, *_args, **_kwargs):
            raise fetcher.httpx.ConnectError("connect failed")

    monkeypatch.setattr(fetcher.httpx, "Client", Client)

    with pytest.raises(PageFetchError, match="request failed"):
        fetch_public_html("https://example.com/product")


def test_fetch_public_html_pins_validated_dns_during_request(monkeypatch) -> None:
    lookups: list[str] = []

    def rebinding_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        lookups.append(str(host))
        address = "93.184.216.34" if len(lookups) <= 2 else "127.0.0.1"
        return [(fetcher.socket.AF_INET, fetcher.socket.SOCK_STREAM, fetcher.socket.IPPROTO_TCP, "", (address, port))]

    class Response:
        status_code = 200
        headers = {"content-type": "text/html"}
        encoding = "utf-8"
        url = "https://rebind.example/product"

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def iter_bytes(self):
            yield PRODUCT_HTML.encode("utf-8")

    class Client:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def stream(self, *_args, **_kwargs):
            resolved = fetcher.socket.getaddrinfo("rebind.example", 443)[0][4][0]
            assert resolved == "93.184.216.34"
            return Response()

    monkeypatch.setattr(fetcher.socket, "getaddrinfo", rebinding_getaddrinfo)
    monkeypatch.setattr(fetcher.httpx, "Client", Client)

    result = fetch_public_html("https://rebind.example/product")

    assert result.final_url == "https://rebind.example/product"


def test_analyze_product_url_uses_render_fallback(monkeypatch) -> None:
    class Page:
        def __init__(self, html: str, mode: str) -> None:
            self.html = html
            self.final_url = "https://example.com/product/charger"
            self.mode = mode

    monkeypatch.setattr(
        page_service,
        "fetch_public_html",
        lambda _url: Page("<html><title>Loading</title></html>", "http"),
    )
    monkeypatch.setattr(page_service, "settings", type("Settings", (), {"enable_rendered_fetch": True})())
    monkeypatch.setattr(
        page_service,
        "render_public_html",
        lambda _url: Page(PRODUCT_HTML, "rendered"),
    )

    result = page_service.analyze_product_url("https://example.com/product/charger")

    assert isinstance(result, ProductPageAnalysis)
    assert result.fetch_mode == "rendered"
    assert result.product.product_title == "Portable Charger"


def test_analyze_product_url_skips_render_fallback_by_default(monkeypatch) -> None:
    monkeypatch.setattr(
        page_service,
        "fetch_public_html",
        lambda _url: type(
            "Page",
            (),
            {
                "html": "<html><title>Loading</title></html>",
                "final_url": "https://example.com/product/charger",
                "mode": "http",
            },
        )(),
    )

    def fail_if_called(_url: str):
        raise AssertionError("render fallback should be disabled by default")

    monkeypatch.setattr(page_service, "settings", type("Settings", (), {"enable_rendered_fetch": False})())
    monkeypatch.setattr(page_service, "render_public_html", fail_if_called)

    with pytest.raises(page_service.ProductNotDetectedError):
        page_service.analyze_product_url("https://example.com/product/charger")


def test_analyze_product_url_uses_render_fallback_when_static_fetch_fails(monkeypatch) -> None:
    class Page:
        def __init__(self, html: str, mode: str) -> None:
            self.html = html
            self.final_url = "https://example.com/product/charger"
            self.mode = mode

    def fail_static_fetch(_url: str):
        raise PageFetchError("Product page returned HTTP status 403.")

    monkeypatch.setattr(page_service, "settings", type("Settings", (), {"enable_rendered_fetch": True})())
    monkeypatch.setattr(page_service, "fetch_public_html", fail_static_fetch)
    monkeypatch.setattr(
        page_service,
        "render_public_html",
        lambda _url: Page(PRODUCT_HTML, "rendered"),
    )

    result = page_service.analyze_product_url("https://example.com/product/charger")

    assert isinstance(result, ProductPageAnalysis)
    assert result.fetch_mode == "rendered"
    assert result.product.product_title == "Portable Charger"


def test_analyze_product_url_reports_both_static_and_render_failures(monkeypatch) -> None:
    def fail_static_fetch(_url: str):
        raise PageFetchError("Product page returned HTTP status 403.")

    def fail_render_fetch(_url: str):
        raise PageFetchError("Rendered page fallback is not installed.")

    monkeypatch.setattr(page_service, "settings", type("Settings", (), {"enable_rendered_fetch": True})())
    monkeypatch.setattr(page_service, "fetch_public_html", fail_static_fetch)
    monkeypatch.setattr(page_service, "render_public_html", fail_render_fetch)

    with pytest.raises(PageFetchError, match="Static fetch failed"):
        page_service.analyze_product_url("https://example.com/product/charger")
