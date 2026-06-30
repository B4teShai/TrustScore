import json

import httpx

from app.schemas.product_analysis import ProductPageData
from app.services.market_reference import FrankfurterExchangeRateProvider, SerperMarketReferenceProvider


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
