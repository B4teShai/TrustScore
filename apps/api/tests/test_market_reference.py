import json

import httpx

from app.schemas.product_analysis import ProductPageData
from app.services.market_reference import (
    FrankfurterExchangeRateProvider,
    SerperMarketReferenceProvider,
    _category_query,
    _similar_enough,
    _tokens,
)

MIULEE_TITLE = (
    "MIULEE 100% Blackout Linen Textured Curtains for Bedroom Solid Thermal Insulated "
    "Natural Beige Grommet Room Darkening Curtains & Drapes Luxury Decor for Living Room "
    "52 x 84 Inch (2 Panels)"
)
CURTAIN_BREADCRUMBS = [
    "Home & Kitchen",
    "Home Décor Products",
    "Window Treatments",
    "Curtains & Drapes",
    "Panels",
]


def test_serper_provider_returns_same_currency_median_and_uses_cache() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.headers["x-api-key"] == "test-key"
        return httpx.Response(
            200,
            json={
                "shopping": [
                    {
                        "title": "Soccer Ball Engraved Glasses Gift",
                        "link": "https://shop.example.com/a",
                        "price": "$19.99",
                    },
                    {
                        "title": "Custom Soccer Ball Engraved Glasses",
                        "link": "https://shop.example.com/b",
                        "price": "USD 25.00",
                    },
                    {
                        "title": "Personalized Soccer Ball Engraved Glasses",
                        "link": "https://shop.example.com/c",
                        "price": "$29.99",
                    },
                    {
                        "title": "Unrelated Baseball Mug",
                        "link": "https://shop.example.com/d",
                        "price": "$9.99",
                    },
                    {
                        "title": "Soccer Ball Engraved Glasses",
                        "link": "https://shop.example.com/e",
                        "price": "MNT 164,690.19",
                    },
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = SerperMarketReferenceProvider(
        api_key="test-key",
        api_url="https://google.serper.dev/shopping",
        cache_ttl_seconds=600,
        http_client=client,
        min_results=3,
    )
    product = ProductPageData(
        url="https://www.etsy.com/listing/1566610967/soccer-ball-engraved-glasses",
        site="www.etsy.com",
        product_title="Soccer Ball Engraved Glasses",
        price=24.5,
        currency="USD",
        reviews=[],
    )

    first = provider.lookup(product, target_market="US")
    second = provider.lookup(product, target_market="US")

    assert first is not None
    assert first.currency == "USD"
    assert first.comparable_count == 3
    assert first.median_price == 25.0
    assert second == first
    assert calls == 1


def test_serper_provider_converts_usd_market_reference_to_jpy_product_currency() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.frankfurter.dev":
            assert request.url.path == "/v2/rate/USD/JPY"
            return httpx.Response(
                200,
                json={"amount": 1, "base": "USD", "quote": "JPY", "date": "2026-06-30", "rate": 156.0},
            )

        body = json.loads(request.content)
        assert body["gl"] == "us"
        assert body["hl"] == "en"
        return httpx.Response(
            200,
            json={
                "shopping": [
                    {
                        "title": "Straight Leg Jeans for Women High Waisted Wide Leg",
                        "link": "https://shop.example.com/a",
                        "price": "$35.99",
                    },
                    {
                        "title": "Straight Leg Jeans Women Loose Stretchy Denim Pants",
                        "link": "https://shop.example.com/b",
                        "price": "USD 39.99",
                    },
                    {
                        "title": "High Waisted Wide Leg Jeans for Women",
                        "link": "https://shop.example.com/c",
                        "price": "$45.00",
                    },
                    {
                        "title": "Straight Leg Jeans for Women",
                        "link": "https://shop.example.com/d",
                        "price": "MNT 164,690.19",
                    },
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    exchange = FrankfurterExchangeRateProvider(
        api_url="https://api.frankfurter.dev/v2",
        cache_ttl_seconds=0,
        http_client=client,
    )
    provider = SerperMarketReferenceProvider(
        api_key="test-key",
        api_url="https://google.serper.dev/shopping",
        cache_ttl_seconds=0,
        http_client=client,
        min_results=3,
        exchange_rate_provider=exchange,
    )
    product = ProductPageData(
        url="https://www.amazon.com/dp/B0TESTUSJP",
        site="www.amazon.com",
        product_title="Straight Leg Jeans for Women High Waisted Wide Leg",
        price=5682,
        currency="JPY",
        reviews=[],
    )

    reference = provider.lookup(product, target_market="US")

    assert reference is not None
    assert reference.currency == "JPY"
    assert reference.comparable_count == 3
    assert reference.median_price == 6238
    assert reference.original_currency == "USD"
    assert reference.exchange_rate == 156
    assert reference.exchange_rate_source == "Frankfurter"
    assert reference.exchange_rate_date == "2026-06-30"


def test_serper_provider_converts_usd_market_reference_to_eur_product_currency() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.frankfurter.dev":
            assert request.url.path == "/v2/rate/USD/EUR"
            return httpx.Response(
                200,
                json={"amount": 1, "base": "USD", "quote": "EUR", "date": "2026-07-01", "rate": 0.88},
            )

        body = json.loads(request.content)
        assert body["gl"] == "fr"
        assert body["hl"] == "fr"
        return httpx.Response(
            200,
            json={
                "shopping": [
                    {
                        "title": "StarTech DisplayPort 1.4 Cable VESA Certified 1m",
                        "link": "https://shop.example.com/a",
                        "price": "$21.99",
                    },
                    {
                        "title": "StarTech DisplayPort Cable 1m DP14VMM1M",
                        "link": "https://shop.example.com/b",
                        "price": "USD 24.99",
                    },
                    {
                        "title": "StarTech.com DisplayPort 1.4 Cable 8K 60Hz",
                        "link": "https://shop.example.com/c",
                        "price": "$29.99",
                    },
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    exchange = FrankfurterExchangeRateProvider(
        api_url="https://api.frankfurter.dev/v2",
        cache_ttl_seconds=0,
        http_client=client,
    )
    provider = SerperMarketReferenceProvider(
        api_key="test-key",
        api_url="https://google.serper.dev/shopping",
        cache_ttl_seconds=0,
        http_client=client,
        min_results=3,
        exchange_rate_provider=exchange,
    )
    product = ProductPageData(
        url="https://www.amazon.fr/dp/B09FP39Q81",
        site="www.amazon.fr",
        product_title="StarTech DisplayPort 1.4 Cable VESA Certified 1m",
        price=23.34,
        currency="EUR",
        reviews=[],
    )

    reference = provider.lookup(product, target_market="EU", locale="fr-FR")

    assert reference is not None
    assert reference.currency == "EUR"
    assert reference.comparable_count == 3
    assert reference.median_price == 21.99
    assert reference.original_currency == "USD"
    assert reference.exchange_rate == 0.88
    assert reference.exchange_rate_source == "Frankfurter"
    assert reference.exchange_rate_date == "2026-07-01"


def test_category_query_uses_breadcrumb_leaf_and_attributes() -> None:
    query = _category_query(MIULEE_TITLE, CURTAIN_BREADCRUMBS)

    assert query is not None
    assert "MIULEE" in query
    assert "Curtains" in query
    assert "Panels" in query
    assert "52 x 84 Inch" in query
    assert "2 Panels" in query
    assert _category_query(MIULEE_TITLE, None) is None
    assert _category_query(MIULEE_TITLE, []) is None


def test_similarity_requirement_capped_for_long_titles() -> None:
    tokens = _tokens(MIULEE_TITLE)

    assert _similar_enough(tokens, "Blackout Curtains 52x84 Inch 2 Panels Beige Grommet")
    assert not _similar_enough(tokens, "Unrelated Baseball Mug")


def test_similarity_matches_japanese_listing_titles() -> None:
    tokens = _tokens("すのこベッド シングル 宮付き ベッドフレーム")

    assert _similar_enough(tokens, "すのこベッド シングル ベッドフレーム 木製")
    assert not _similar_enough(tokens, "コーヒーメーカー 全自動")


def test_lookup_retries_with_category_query_and_caches_combined_result() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body["q"])
        if "Curtains" in body["q"] and "Blackout" not in body["q"]:
            return httpx.Response(
                200,
                json={
                    "shopping": [
                        {
                            "title": "Blackout Curtains 52x84 Inch 2 Panels Beige",
                            "link": "https://shop.example.com/a",
                            "price": "$21.99",
                        },
                        {
                            "title": "MIULEE Curtains Room Darkening Panels",
                            "link": "https://shop.example.com/b",
                            "price": "$24.99",
                        },
                        {
                            "title": "Linen Textured Curtains Drapes 2 Panels",
                            "link": "https://shop.example.com/c",
                            "price": "$27.99",
                        },
                    ]
                },
            )
        return httpx.Response(
            200,
            json={
                "shopping": [
                    {
                        "title": "Unrelated Baseball Mug",
                        "link": "https://shop.example.com/x",
                        "price": "$9.99",
                    }
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = SerperMarketReferenceProvider(
        api_key="test-key",
        api_url="https://google.serper.dev/shopping",
        cache_ttl_seconds=600,
        http_client=client,
        min_results=3,
    )
    product = ProductPageData(
        url="https://www.amazon.com/dp/B08SBYPF14",
        site="www.amazon.com",
        product_title=MIULEE_TITLE[:240],
        price=24.39,
        currency="USD",
        category_path=CURTAIN_BREADCRUMBS,
        reviews=[],
    )

    first = provider.lookup(product, target_market="US")
    second = provider.lookup(product, target_market="US")

    assert first is not None
    assert first.median_price == 24.99
    assert first.comparable_count == 3
    assert first.query_strategy == "category"
    assert second == first
    assert len(calls) == 2


def test_lookup_makes_single_call_when_primary_query_succeeds() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={
                "shopping": [
                    {
                        "title": "MIULEE Blackout Linen Textured Curtains Beige",
                        "link": "https://shop.example.com/a",
                        "price": "$21.99",
                    },
                    {
                        "title": "Blackout Linen Curtains Thermal Insulated Grommet Beige",
                        "link": "https://shop.example.com/b",
                        "price": "$24.99",
                    },
                    {
                        "title": "Linen Textured Blackout Curtains Room Darkening Bedroom",
                        "link": "https://shop.example.com/c",
                        "price": "$27.99",
                    },
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = SerperMarketReferenceProvider(
        api_key="test-key",
        api_url="https://google.serper.dev/shopping",
        cache_ttl_seconds=0,
        http_client=client,
        min_results=3,
    )
    product = ProductPageData(
        url="https://www.amazon.com/dp/B08SBYPF14",
        site="www.amazon.com",
        product_title=MIULEE_TITLE[:240],
        price=24.39,
        currency="USD",
        category_path=CURTAIN_BREADCRUMBS,
        reviews=[],
    )

    reference = provider.lookup(product, target_market="US")

    assert reference is not None
    assert reference.query_strategy == "title"
    assert calls == 1


def test_jp_market_parses_yen_prices_and_japanese_titles() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["gl"] == "jp"
        assert body["hl"] == "ja"
        return httpx.Response(
            200,
            json={
                "shopping": [
                    {
                        "title": "すのこベッド シングル ベッドフレーム 宮付き",
                        "link": "https://shop.example.jp/a",
                        "price": "¥12,980",
                    },
                    {
                        "title": "ベッドフレーム シングル すのこ 木製",
                        "link": "https://shop.example.jp/b",
                        "price": "13980円",
                    },
                    {
                        "title": "宮付き すのこベッド フレーム シングル",
                        "link": "https://shop.example.jp/c",
                        "price": "JPY 15,800",
                    },
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = SerperMarketReferenceProvider(
        api_key="test-key",
        api_url="https://google.serper.dev/shopping",
        cache_ttl_seconds=0,
        http_client=client,
        min_results=3,
    )
    product = ProductPageData(
        url="https://www.amazon.co.jp/-/en/dp/B0FC247FC1",
        site="www.amazon.co.jp",
        product_title="すのこベッド シングル 宮付き ベッドフレーム",
        price=13500,
        currency="JPY",
        reviews=[],
    )

    reference = provider.lookup(product, target_market="JP")

    assert reference is not None
    assert reference.currency == "JPY"
    assert reference.comparable_count == 3
    assert reference.median_price == 13980


def test_category_fallback_accepts_cross_script_listings_and_trims_outliers() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body["q"])
        if "Bed" in body["q"] and "Soft Pad" not in body["q"]:
            return httpx.Response(
                200,
                json={
                    "shopping": [
                        {
                            "title": "ファブリックベッドフレーム シングル",
                            "link": "https://shop.example.jp/a",
                            "price": "￥21,032",
                        },
                        {
                            "title": "布張り ベッドフレーム すのこ 静音",
                            "link": "https://shop.example.jp/b",
                            "price": "￥19,990",
                        },
                        {
                            "title": "ベッドフレーム シングル ファブリック 北欧",
                            "link": "https://shop.example.jp/c",
                            "price": "￥24,800",
                        },
                        {
                            "title": "Luxury Imported King Upholstered Panel Bed",
                            "link": "https://shop.example.jp/d",
                            "price": "￥626,270",
                        },
                    ]
                },
            )
        return httpx.Response(200, json={"shopping": []})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = SerperMarketReferenceProvider(
        api_key="test-key",
        api_url="https://google.serper.dev/shopping",
        cache_ttl_seconds=0,
        http_client=client,
        min_results=3,
    )
    product = ProductPageData(
        url="https://www.amazon.co.jp/-/en/dp/B0FC247FC1",
        site="www.amazon.co.jp",
        product_title="Upholstered Bed Frame with Soft Pad Inclined Headboard Fabric Bed Single",
        price=19800,
        currency="JPY",
        category_path=["Home & Kitchen", "Furniture", "Beds, Frames & Bases", "Bed Frames"],
        reviews=[],
    )

    reference = provider.lookup(product, target_market="JP")

    assert reference is not None
    assert reference.query_strategy == "category"
    assert reference.comparable_count == 3
    assert reference.median_price == 21032
    assert len(calls) == 2
