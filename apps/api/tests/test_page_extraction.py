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


def test_extract_product_page_rejects_amazon_header_returns_as_policy() -> None:
    html = """
    <html>
      <body>
        <h1>Natural Burlap Placemats</h1>
        <nav>Hello, sign in Account & Lists Returns & Orders 0 Cart All Today's Deals</nav>
        <span>$39.99</span>
        <button>Add to cart</button>
      </body>
    </html>
    """

    result = extract_product_page(html, "https://www.amazon.com/dp/B0TEST1234")

    assert result.detected is True
    assert result.product is not None
    assert result.product.return_policy is None
    assert "policy_rejected_noisy_text" in result.signals


def test_extract_product_page_accepts_real_return_policy() -> None:
    html = """
    <html>
      <body>
        <h1>Natural Burlap Placemats</h1>
        <span>$39.99</span>
        <button>Add to cart</button>
        <p>This item is eligible for free returns within 30 days of delivery.</p>
      </body>
    </html>
    """

    result = extract_product_page(html, "https://www.amazon.com/dp/B0TEST1234")

    assert result.detected is True
    assert result.product is not None
    assert result.product.return_policy == (
        "This item is eligible for free returns within 30 days of delivery."
    )
    assert "policy" in result.signals


def test_extract_product_page_strips_marketplace_review_boilerplate() -> None:
    html = """
    <html>
      <body>
        <h1>Natural Burlap Placemats</h1>
        <span>$39.99</span>
        <button>Add to cart</button>
        <div data-hook="review">
          <span data-hook="reviewText">
            Brief content visible, double tap to read full content.
            Full content visible, double tap to read brief content.
            Very high quality and durable outside.
            Read more Read less
          </span>
        </div>
      </body>
    </html>
    """

    result = extract_product_page(html, "https://www.amazon.com/dp/B0TEST1234")

    assert result.detected is True
    assert result.product is not None
    assert result.product.reviews[0].text == "Very high quality and durable outside."


def test_extract_product_page_truncates_title_at_word_boundary() -> None:
    long_title = ("Round Natural Burlap Placemat Charger " * 10).strip()
    html = f"""
    <html>
      <body>
        <h1>{long_title}</h1>
        <span>$39.99</span>
        <button>Add to cart</button>
      </body>
    </html>
    """

    result = extract_product_page(html, "https://www.amazon.com/dp/B0TEST1234")

    assert result.detected is True
    assert result.product is not None
    assert len(result.product.product_title) <= 240
    assert set(result.product.product_title.split()) <= {
        "Round",
        "Natural",
        "Burlap",
        "Placemat",
        "Charger",
    }


def test_extract_product_page_from_ebay_like_markup() -> None:
    html = """
    <html>
      <body>
        <h1 class="x-item-title__mainTitle"><span>Refurbished Noise Cancelling Headphones</span></h1>
        <div data-testid="x-price-primary"><span>US $49.99</span></div>
        <img id="icImg" src="https://i.ebayimg.com/images/g/headphones/s-l1600.jpg" />
        <section data-testid="x-sellercard-atf__info">
          <a>trusted_audio_shop</a>
        </section>
        <div data-testid="x-sellercard-atf__data-item">
          99.2% positive feedback
        </div>
        <div data-testid="x-sellercard-atf__data-item">
          12,345 feedback
        </div>
        <div data-testid="x-returns-minview">30 day returns. Buyer pays for return shipping.</div>
        <button>Buy It Now</button>
      </body>
    </html>
    """

    result = extract_product_page(html, "https://www.ebay.com/itm/188272417479")

    assert result.detected is True
    assert result.product is not None
    assert "site_ebay" in result.signals
    assert result.product.product_title == "Refurbished Noise Cancelling Headphones"
    assert result.product.price == 49.99
    assert result.product.currency == "USD"
    assert result.product.product_image_url == "https://i.ebayimg.com/images/g/headphones/s-l1600.jpg"
    assert result.product.seller is not None
    assert result.product.seller.name == "trusted_audio_shop"
    assert result.product.seller.rating == 5.0
    assert result.product.seller.review_count == 12345
    assert result.product.return_policy == "30 day returns."


def test_extract_product_page_from_etsy_like_markup() -> None:
    html = """
    <html>
      <body>
        <h1 data-buy-box-listing-title>Soccer Ball Engraved Glasses</h1>
        <p data-buy-box-region="price">$24.50</p>
        <img src="https://i.etsystatic.com/123/listing.jpg" />
        <a data-buy-box-region="shop-name" href="/shop/GoalGiftShop">GoalGiftShop</a>
        <div data-region="shop-rating">4.9 stars 1,284 reviews</div>
        <div data-policies-return-policy>Returns accepted within 30 days.</div>
        <div data-review-region>
          <p>Beautiful glass and clean engraving for a soccer gift.</p>
        </div>
        <button>Add to cart</button>
      </body>
    </html>
    """

    result = extract_product_page(
        html,
        "https://www.etsy.com/listing/1566610967/soccer-ball-engraved-glasses",
    )

    assert result.detected is True
    assert result.product is not None
    assert "site_etsy" in result.signals
    assert result.product.product_title == "Soccer Ball Engraved Glasses"
    assert result.product.price == 24.50
    assert result.product.currency == "USD"
    assert result.product.seller is not None
    assert result.product.seller.name == "GoalGiftShop"
    assert result.product.seller.rating == 4.9
    assert result.product.seller.review_count == 1284
    assert result.product.return_policy == "Returns accepted within 30 days."
    assert result.product.reviews[0].text == "Beautiful glass and clean engraving for a soccer gift."


def test_extract_product_page_from_bestbuy_product_markup() -> None:
    html = """
    <html>
      <head>
        <meta property="og:title" content="Logitech MX Vertical Mouse" />
        <meta property="og:image" content="https://pisces.bbystatic.com/image2.jpg" />
      </head>
      <body>
        <h1 data-testid="product-title">Logitech MX Vertical Mouse</h1>
        <div data-testid="customer-price">$99.99</div>
        <div data-testid="brand-name">Logitech</div>
        <div data-testid="ratings-and-reviews">4.6 out of 5 stars with 2,345 reviews</div>
        <div data-testid="product-description">Ergonomic wireless mouse.</div>
        <div data-testid="return-policy">15-day return policy available.</div>
        <button>Add to Cart</button>
      </body>
    </html>
    """

    result = extract_product_page(
        html,
        "https://www.bestbuy.com/site/logitech-mx-vertical-mouse/6282602.p",
    )

    assert result.detected is True
    assert result.page_type == "product"
    assert result.product is not None
    assert "site_bestbuy" in result.signals
    assert result.product.product_title == "Logitech MX Vertical Mouse"
    assert result.product.price == 99.99
    assert result.product.currency == "USD"
    assert result.product.product_image_url == "https://pisces.bbystatic.com/image2.jpg"
    assert result.product.review_count == 2345
    assert result.product.return_policy == "15-day return policy available."


def test_extract_product_page_rejects_bestbuy_reviews_page_without_canonical_product() -> None:
    html = """
    <html>
      <body>
        <h1>Customer Ratings & Reviews</h1>
        <div class="review-item"><p>Comfortable mouse after long use.</p></div>
        <div aria-label="4.5 out of 5 stars">4.5 stars</div>
      </body>
    </html>
    """

    result = extract_product_page(
        html,
        "https://www.bestbuy.com/site/reviews/logitech-mx-vertical-mouse/6282602",
    )

    assert result.detected is False
    assert result.product is None
    assert result.page_type == "review_page"


def test_extract_product_page_canonicalizes_bestbuy_reviews_page_with_product_link() -> None:
    html = """
    <html>
      <body>
        <a href="/site/logitech-mx-vertical-mouse/6282602.p">View product</a>
        <h1 data-testid="product-title">Logitech MX Vertical Mouse</h1>
        <div data-testid="customer-price">$99.99</div>
        <div data-testid="brand-name">Logitech</div>
        <button>Add to Cart</button>
      </body>
    </html>
    """

    result = extract_product_page(
        html,
        "https://www.bestbuy.com/site/reviews/logitech-mx-vertical-mouse/6282602",
    )

    assert result.detected is True
    assert result.page_type == "review_page"
    assert result.canonical_product_url == "https://www.bestbuy.com/site/logitech-mx-vertical-mouse/6282602.p"
    assert result.product is not None
    assert result.product.url == "https://www.bestbuy.com/site/logitech-mx-vertical-mouse/6282602.p"


def test_extract_product_page_parses_suffix_currency_generic_markup() -> None:
    html = """
    <html>
      <body>
        <h1>Desk Organizer</h1>
        <span class="price">19.99 USD</span>
        <button>Add to cart</button>
      </body>
    </html>
    """

    result = extract_product_page(html, "https://shop.example.com/products/desk-organizer")

    assert result.detected is True
    assert result.product is not None
    assert result.product.price == 19.99
    assert result.product.currency == "USD"


def test_extract_product_page_ignores_localized_marketplace_currency() -> None:
    html = """
    <html>
      <body>
        <h1>Natural Burlap Placemats</h1>
        <span class="price">MNT 164,690.19</span>
        <div class="seller">Sold by Example Store</div>
        <button>Add to cart</button>
      </body>
    </html>
    """

    result = extract_product_page(
        html,
        "https://www.amazon.com/dp/B0GXVNG3TR",
        target_market="US",
    )

    assert result.detected is True
    assert result.product is not None
    assert result.product.price is None
    assert result.product.currency is None
    assert "price_ignored_localized_currency:MNT" in result.signals


def test_extract_product_page_reads_visible_amazon_jp_price_block() -> None:
    html = """
    <html>
      <body>
        <h1>Japanese Snack Box</h1>
        <div id="corePrice_feature_div">
          <span class="a-price">JPY2,659 JPY&nbsp;221 per count(JPY221 / count)</span>
        </div>
        <div id="merchant-info">Sold by Example Japan Store</div>
        <button>Add to cart</button>
      </body>
    </html>
    """

    result = extract_product_page(html, "https://www.amazon.co.jp/dp/B0TESTJP12")

    assert result.detected is True
    assert result.product is not None
    assert result.product.price == 2659
    assert result.product.currency == "JPY"
    assert "price" in result.signals
    assert "price_ignored_localized_currency:JPY" not in result.signals


def test_extract_product_page_reads_split_amazon_jp_price() -> None:
    html = """
    <html>
      <body>
        <h1>Magic Trackpad</h1>
        <div id="corePriceDisplay_desktop_feature_div">
          <span class="a-price-symbol">￥</span>
          <span class="a-price-whole">16,800</span>
        </div>
        <div id="bylineInfo">Visit the Apple Store</div>
        <button>Add to cart</button>
      </body>
    </html>
    """

    result = extract_product_page(html, "https://www.amazon.co.jp/dp/B0TESTJP13")

    assert result.detected is True
    assert result.product is not None
    assert result.product.price == 16800
    assert result.product.currency == "JPY"


def test_extract_product_page_prefers_amazon_jp_buying_option_over_sponsored_price() -> None:
    html = """
    <html>
      <body>
        <h1>Yamazen EDC-H601(B) Dehumidifier</h1>
        <div class="sponsored-carousel">
          <span class="a-price">¥8,628</span>
        </div>
        <div id="centerCol">
          <p>No featured offers available</p>
          <p>gray (dark gray) 3 options from ¥15,800</p>
          <p>white 2 options from ¥15,800</p>
          <p>gray (light gray) 3 options from ¥15,100</p>
        </div>
        <div id="bylineInfo">Visit the 山善(YAMAZEN) Store</div>
        <button>Add to cart</button>
      </body>
    </html>
    """

    result = extract_product_page(html, "https://www.amazon.co.jp/dp/B0TESTJP14")

    assert result.detected is True
    assert result.product is not None
    assert result.product.price == 15800
    assert result.product.currency == "JPY"


def test_extract_product_page_accepts_japanese_return_policy() -> None:
    html = """
    <html>
      <body>
        <h1>Magic Trackpad</h1>
        <span>￥16,800</span>
        <div id="bylineInfo">Visit the Apple Store</div>
        <p>この商品は30日以内の返品と返金に対応しています。</p>
        <button>Add to cart</button>
      </body>
    </html>
    """

    result = extract_product_page(html, "https://www.amazon.co.jp/dp/B0TESTJP14")

    assert result.detected is True
    assert result.product is not None
    assert result.product.return_policy == "この商品は30日以内の返品と返金に対応しています。"
    assert "policy" in result.signals


def test_extract_product_page_accepts_french_return_policy() -> None:
    html = """
    <html>
      <body>
        <h1>Cable DisplayPort StarTech</h1>
        <span>€23.34</span>
        <div id="bylineInfo">Visit the StarTech Store</div>
        <p>Retour et remboursement possibles sous 30 jours avec garantie.</p>
        <button>Add to cart</button>
      </body>
    </html>
    """

    result = extract_product_page(html, "https://www.amazon.fr/dp/B0TESTFR12")

    assert result.detected is True
    assert result.product is not None
    assert result.product.return_policy == "Retour et remboursement possibles sous 30 jours avec garantie."
    assert "policy" in result.signals


def test_extract_product_page_keeps_supported_jpy_price_on_amazon_us() -> None:
    html = """
    <html>
      <body>
        <h1>Straight Leg Jeans for Women</h1>
        <div id="corePrice_feature_div">
          <span class="a-price">JPY5,682</span>
        </div>
        <div id="merchant-info">Sold by Mars power</div>
        <button>Add to cart</button>
      </body>
    </html>
    """

    result = extract_product_page(
        html,
        "https://www.amazon.com/dp/B0TESTUSJP",
        target_market="US",
    )

    assert result.detected is True
    assert result.product is not None
    assert result.product.price == 5682
    assert result.product.currency == "JPY"
    assert "price" in result.signals
    assert "price_ignored_localized_currency:JPY" not in result.signals


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
