from fastapi.testclient import TestClient
import pytest

from app.api import product_analysis as product_routes
from app.main import app
from app.page_fetching.fetcher import PageFetchError
from app.schemas.product_analysis import ProductPageData, ReviewInput, SellerInfo
from app.services.product_page_analysis import ProductPageAnalysis, ProductNotDetectedError


@pytest.fixture(autouse=True)
def _disable_live_market_reference(monkeypatch) -> None:
    monkeypatch.setattr(product_routes, "enrich_market_reference", _no_market_reference)


def _no_market_reference(product: ProductPageData, **_kwargs) -> tuple[ProductPageData, list[str]]:
    return product, []


def _expected_risk_level(score: int) -> str:
    if score >= 80:
        return "Low Risk"
    if score >= 50:
        return "Medium Risk"
    return "High Risk"


def _analysis_for(url: str, **_kwargs) -> ProductPageAnalysis:
    return ProductPageAnalysis(
        product=ProductPageData(
            url=url,
            site="example.com",
            product_title="Wireless Headphones",
            product_image_url="https://example.com/product-image.jpg",
            price=29.99,
            currency="USD",
            average_market_price=59.99,
            seller=SellerInfo(name="Example Store", rating=4.2, review_count=980),
            return_policy="30-day return policy available",
            reviews=[
                ReviewInput(text="Great product and fast delivery.", rating=5),
                ReviewInput(text="Bad quality, broke after one day.", rating=1),
            ],
            rating=4.1,
            review_count=230,
        ),
        fetch_mode="http",
        signals=["structured_product", "price", "seller"],
    )


def test_analyze_product_accepts_url_only_and_returns_product(monkeypatch) -> None:
    client = TestClient(app)
    monkeypatch.setattr(product_routes, "analyze_product_url", _analysis_for)

    response = client.post(
        "/api/analyze-product",
        json={"url": "https://example.com/product/123"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["scan_id"]
    assert body["product"]["url"] == "https://example.com/product/123"
    assert body["product"]["product_title"] == "Wireless Headphones"
    assert body["product"]["seller_name"] == "Example Store"
    assert 0 <= body["trust_score"] <= 100
    assert body["risk_level"] == _expected_risk_level(body["trust_score"])
    assert 0 <= body["confidence"] <= 1
    assert len(body["top_reasons"]) == 3
    assert body["is_mock"] is False
    assert body["model_version"] == "0.3.0"
    assert body["fetch_mode"] == "http"
    assert body["extraction_signals"] == ["structured_product", "price", "seller"]
    assert body["model_modes"]["user_feedback_history"] == "not_applied"
    assert body["score_semantics"]
    assert body["evidence"]
    assert isinstance(body["missing_inputs"], list)
    assert all(0 <= score <= 100 for score in body["component_scores"].values())


def test_scan_v1_returns_url_only_inference_response(monkeypatch) -> None:
    client = TestClient(app)
    monkeypatch.setattr(product_routes, "analyze_product_url", _analysis_for)

    response = client.post(
        "/api/v1/scan",
        json={"url": "https://example.com/product/123"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["is_mock"] is False
    assert body["product"]["product_title"] == "Wireless Headphones"
    assert 0 <= body["component_scores"]["price_safety"] <= 100


def test_scan_extracted_scores_active_tab_product_data() -> None:
    client = TestClient(app)

    response = client.post(
        "/api/v1/scan-extracted",
        json={
            "product": {
                "url": "https://example.com/product/123",
                "site": "example.com",
                "product_title": "Amazon Essentials Women's Mid-Rise Stretchy Skinny Jeans",
                "product_image_url": "https://example.com/jeans.jpg",
                "price": 16.4,
                "currency": "USD",
                "seller": {"name": "Amazon Essentials Store"},
                "rating": 4.3,
                "review_count": 18806,
                "reviews": [],
            }
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["fetch_mode"] == "extension_dom"
    assert "extension_dom" in body["extraction_signals"]
    assert body["product"]["product_title"].startswith("Amazon Essentials")
    assert body["product"]["seller_name"] == "Amazon Essentials Store"
    assert body["missing_inputs"] == ["visible review text"]
    assert body["model_modes"]["price_safety"] == "not_scored_missing_market_reference"
    assert body["score_status"] in {"scored", "low_evidence_triage"}
    assert body["page_type"] == "product"
    assert body["market_context"]["resolved_market"] == "US"
    assert body["score_trace"]
    evidence = {item["component"]: item for item in body["evidence"]}
    assert "No verified market reference found." in evidence["price_safety"]["evidence"]


def test_scan_extracted_uses_market_reference_when_serper_finds_comparables(monkeypatch) -> None:
    client = TestClient(app)

    def fake_market_reference(
        product: ProductPageData,
        **_kwargs,
    ) -> tuple[ProductPageData, list[str]]:
        return (
            product.model_copy(
                update={
                    "average_market_price": 25.0,
                    "market_reference_count": 12,
                    "market_reference_source": "Serper",
                }
            ),
            ["market_reference:serper:count=12"],
        )

    monkeypatch.setattr(product_routes, "enrich_market_reference", fake_market_reference)

    response = client.post(
        "/api/v1/scan-extracted",
        json={
            "product": {
                "url": "https://www.etsy.com/listing/1566610967/soccer-ball-engraved-glasses",
                "site": "www.etsy.com",
                "product_title": "Soccer Ball Engraved Glasses",
                "price": 24.5,
                "currency": "USD",
                "seller": {"name": "GoalGiftShop", "rating": 4.9, "review_count": 1284},
                "return_policy": "Returns accepted within 30 days.",
                "reviews": [{"text": "Great engraving quality."}],
            }
        },
    )

    assert response.status_code == 200
    body = response.json()
    evidence = {item["component"]: item for item in body["evidence"]}
    assert body["model_modes"]["price_safety"] != "not_scored_missing_market_reference"
    assert "market_reference:serper:count=12" in body["extraction_signals"]
    assert evidence["price_safety"]["evidence"] == [
        "Listed price: USD 24.50",
        "Market reference found from 12 comparable listings: USD 25.00 (Serper)",
    ]


def test_scan_extracted_sanitizes_messy_active_tab_preview() -> None:
    client = TestClient(app)

    response = client.post(
        "/api/v1/scan-extracted",
        json={
            "product": {
                "url": "https://www.amazon.co.jp/dp/B0DZC1K3B7?ref="
                + ("a" * 3000),
                "site": "www.amazon.co.jp",
                "product_title": "Japanese product " + ("x" * 500),
                "product_image_url": "http://127.0.0.1/private.png",
                "price": 12000,
                "currency": "JPY plus extra text that should be truncated",
                "seller": {"name": "Seller " + ("y" * 500)},
                "reviews": [],
            }
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["fetch_mode"] == "extension_dom"
    assert len(body["product"]["product_title"]) == 240
    assert body["product"]["product_image_url"] is None
    assert body["product"]["currency"] == "JPY"
    assert len(body["product"]["seller_name"]) == 160


def test_scan_extracted_tolerates_extra_browser_fields_and_bad_optional_ratings() -> None:
    client = TestClient(app)

    response = client.post(
        "/api/v1/scan-extracted",
        json={
            "target_market": "US",
            "unexpected_top_level": "ignored",
            "product": {
                "url": "https://www.ebay.com/itm/188272417479",
                "site": "www.ebay.com",
                "product_title": "Wireless Charging Pad",
                "price": 19.99,
                "currency": "USD",
                "seller": {
                    "name": "trusted_audio_shop",
                    "rating": 98.8,
                    "review_count": 12345,
                    "raw_feedback": "98.8% positive feedback",
                },
                "rating": 98.8,
                "review_count": 120,
                "reviews": [
                    {
                        "text": "Works well.",
                        "rating": 98.8,
                        "raw_node_id": "review-1",
                    }
                ],
                "units_bought_recent": 250,
                "browser_only_field": "ignored",
            },
        },
    )

    assert response.status_code == 200
    body = response.json()
    evidence = {item["component"]: item for item in body["evidence"]}
    assert body["product"]["seller_name"] == "trusted_audio_shop"
    assert "Seller rating" not in " ".join(evidence["seller_reliability"]["evidence"])
    assert evidence["review_authenticity"]["evidence"][0].startswith("1 visible reviews")


def test_scan_extracted_cleans_noisy_policy_and_review_boilerplate() -> None:
    client = TestClient(app)

    response = client.post(
        "/api/v1/scan-extracted",
        json={
            "product": {
                "url": "https://www.amazon.com/dp/B0TEST1234",
                "site": "www.amazon.com",
                "product_title": "Natural Burlap Placemats",
                "price": 39.99,
                "currency": "USD",
                "seller": {"name": "Example Store"},
                "return_policy": (
                    "Hello, sign in Account & Lists Returns & Orders 0 Cart All Today's Deals"
                ),
                "reviews": [
                    {
                        "text": (
                            "Brief content visible, double tap to read full content. "
                            "Full content visible, double tap to read brief content. "
                            "Very high quality and durable outside. Read more Read less"
                        )
                    }
                ],
            }
        },
    )

    assert response.status_code == 200
    body = response.json()
    evidence = {item["component"]: item for item in body["evidence"]}
    assert evidence["review_authenticity"]["evidence"][0].startswith("1 visible reviews")
    assert evidence["return_policy_clarity"]["evidence"] == [
        "Return policy not visible on this page."
    ]
    assert evidence["return_policy_clarity"]["missing_inputs"] == []
    assert body["missing_inputs"] == []


def test_scan_extracted_ignores_localized_marketplace_currency() -> None:
    client = TestClient(app)

    response = client.post(
        "/api/v1/scan-extracted",
        json={
            "target_market": "US",
            "product": {
                "url": "https://www.amazon.com/dp/B0GXVNG3TR",
                "site": "www.amazon.com",
                "product_title": "Natural Burlap Placemats",
                "price": 164690.19,
                "currency": "MNT",
                "seller": {"name": "Example Store"},
                "rating": 4.5,
                "review_count": 18,
                "reviews": [{"text": "Very high quality and durable outside."}],
            },
        },
    )

    assert response.status_code == 200
    body = response.json()
    evidence = {item["component"]: item for item in body["evidence"]}
    assert body["product"]["price"] is None
    assert body["product"]["currency"] is None
    assert "price_ignored_localized_currency:MNT" in body["extraction_signals"]
    assert evidence["price_safety"]["evidence"] == [
        "Price not used: localized marketplace currency (MNT)"
    ]
    assert body["model_modes"]["price_safety"] == "not_scored_localized_currency:MNT"
    assert "product price" not in body["missing_inputs"]


def test_scan_extracted_does_not_show_market_reference_after_ignored_price(monkeypatch) -> None:
    client = TestClient(app)

    def fake_market_reference(
        product: ProductPageData,
        **kwargs,
    ) -> tuple[ProductPageData, list[str]]:
        assert kwargs.get("allow_without_listed_price") is not True
        assert product.price is None
        return (
            product.model_copy(
                update={
                    "average_market_price": 27.05,
                    "currency": "USD",
                    "market_reference_count": 12,
                    "market_reference_source": "Serper",
                }
            ),
            ["market_reference:serper:count=12"],
        )

    monkeypatch.setattr(product_routes, "enrich_market_reference", fake_market_reference)

    response = client.post(
        "/api/v1/scan-extracted",
        json={
            "target_market": "US",
            "product": {
                "url": "https://www.amazon.com/dp/B0GXVNG3TR",
                "site": "www.amazon.com",
                "product_title": "Natural Burlap Placemats",
                "price": 164690.19,
                "currency": "MNT",
                "seller": {"name": "Example Store"},
                "reviews": [{"text": "Very high quality and durable outside."}],
            },
        },
    )

    assert response.status_code == 200
    body = response.json()
    evidence = {item["component"]: item for item in body["evidence"]}
    assert evidence["price_safety"]["evidence"] == [
        "Price not used: localized marketplace currency (MNT)"
    ]
    assert body["model_modes"]["price_safety"] == "not_scored_localized_currency:MNT"


def test_scan_extracted_keeps_same_market_jpy_price_without_market_reference() -> None:
    client = TestClient(app)

    response = client.post(
        "/api/v1/scan-extracted",
        json={
            "product": {
                "url": "https://www.amazon.co.jp/dp/B0TESTJP12",
                "site": "www.amazon.co.jp",
                "product_title": "Japanese Snack Box",
                "price": 2659,
                "currency": "JPY",
                "seller": {"name": "Example Japan Store"},
                "rating": 4.5,
                "review_count": 18,
                "reviews": [{"text": "Fresh snacks and fast delivery."}],
            },
        },
    )

    assert response.status_code == 200
    body = response.json()
    evidence = {item["component"]: item for item in body["evidence"]}
    assert body["product"]["price"] == 2659
    assert body["product"]["currency"] == "JPY"
    assert body["model_modes"]["price_safety"] == "not_scored_missing_market_reference"
    assert body["market_context"]["resolved_market"] == "JP"
    assert evidence["price_safety"]["evidence"] == [
        "Listed price only: JPY 2,659",
        "No verified market reference found.",
    ]
    assert evidence["price_safety"]["missing_inputs"] == [
        "verified same-currency market reference"
    ]
    assert "product price" not in body["missing_inputs"]


def test_scan_extracted_keeps_supported_jpy_price_on_amazon_us() -> None:
    client = TestClient(app)

    response = client.post(
        "/api/v1/scan-extracted",
        json={
            "target_market": "US",
            "product": {
                "url": "https://www.amazon.com/dp/B0TESTUSJP",
                "site": "www.amazon.com",
                "product_title": "Straight Leg Jeans for Women",
                "price": 5682,
                "currency": "JPY",
                "seller": {"name": "Mars power"},
                "rating": 4.3,
                "review_count": 1037,
                "reviews": [{"text": "Comfortable fit and stretchy denim."}],
            },
        },
    )

    assert response.status_code == 200
    body = response.json()
    evidence = {item["component"]: item for item in body["evidence"]}
    assert body["product"]["price"] == 5682
    assert body["product"]["currency"] == "JPY"
    assert "price_ignored_localized_currency:JPY" not in body["extraction_signals"]
    assert evidence["price_safety"]["evidence"] == [
        "Listed price only: JPY 5,682",
        "No verified market reference found.",
    ]
    assert evidence["price_safety"]["missing_inputs"] == [
        "verified same-currency market reference"
    ]
    assert body["model_modes"]["price_safety"] == "not_scored_missing_market_reference"


def test_scan_extracted_scores_supported_jpy_with_jpy_market_reference(monkeypatch) -> None:
    client = TestClient(app)

    def fake_market_reference(
        product: ProductPageData,
        **_kwargs,
    ) -> tuple[ProductPageData, list[str]]:
        return (
            product.model_copy(
                update={
                    "average_market_price": 5900,
                    "currency": "JPY",
                    "market_reference_count": 3,
                    "market_reference_source": "Serper",
                    "market_reference_original_currency": "USD",
                    "market_reference_exchange_rate": 156,
                    "market_reference_exchange_rate_source": "Frankfurter",
                    "market_reference_exchange_rate_date": "2026-06-30",
                }
            ),
            ["market_reference:serper:count=3"],
        )

    monkeypatch.setattr(product_routes, "enrich_market_reference", fake_market_reference)

    response = client.post(
        "/api/v1/scan-extracted",
        json={
            "target_market": "US",
            "product": {
                "url": "https://www.amazon.com/dp/B0TESTUSJP",
                "site": "www.amazon.com",
                "product_title": "Straight Leg Jeans for Women",
                "price": 5682,
                "currency": "JPY",
                "seller": {"name": "Mars power"},
                "rating": 4.3,
                "review_count": 1037,
                "reviews": [{"text": "Comfortable fit and stretchy denim."}],
            },
        },
    )

    assert response.status_code == 200
    body = response.json()
    evidence = {item["component"]: item for item in body["evidence"]}
    assert body["product"]["price"] == 5682
    assert body["product"]["currency"] == "JPY"
    assert body["model_modes"]["price_safety"] != "not_scored_missing_market_reference"
    assert evidence["price_safety"]["evidence"] == [
        "Listed price: JPY 5,682",
        "Market reference found from 3 comparable listings: JPY 5,900 (Serper)",
        "Converted market reference from USD: 1 USD = JPY 156 (Frankfurter, 2026-06-30)",
    ]
    assert evidence["price_safety"]["missing_inputs"] == []


def test_scan_extracted_scores_japanese_policy_and_official_store() -> None:
    client = TestClient(app)

    response = client.post(
        "/api/v1/scan-extracted",
        json={
            "target_market": "JP",
            "locale": "ja-JP",
            "product": {
                "url": "https://www.amazon.co.jp/dp/B0TESTJP14",
                "site": "www.amazon.co.jp",
                "product_title": "Magic Trackpad (USB-C)",
                "price": 16800,
                "currency": "JPY",
                "seller": {
                    "name": "Apple Store",
                    "brand_store_name": "Apple Store",
                    "is_official_store": True,
                },
                "return_policy": "この商品は30日以内の返品と返金に対応しています。",
                "reviews": [{"text": "反応が良く、品質も高いです。", "rating": 5}],
            },
        },
    )

    assert response.status_code == 200
    body = response.json()
    evidence = {item["component"]: item for item in body["evidence"]}
    assert body["model_modes"]["return_policy_clarity"] != "not_scored_missing_policy"
    assert body["component_scores"]["seller_reliability"] >= 70
    assert any("official brand store" in item for item in evidence["seller_reliability"]["evidence"])


def test_scan_extracted_marks_review_page_as_low_evidence() -> None:
    client = TestClient(app)

    response = client.post(
        "/api/v1/scan-extracted",
        json={
            "product": {
                "url": "https://www.bestbuy.com/site/reviews/logitech-mouse/6282602",
                "site": "www.bestbuy.com",
                "product_title": "Customer Ratings & Reviews",
                "reviews": [{"text": "Comfortable mouse after long use."}],
            },
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["page_type"] == "review_page"
    assert body["score_status"] == "low_evidence_triage"
    assert body["product_identity_confidence"] == 0


def test_scan_returns_product_not_detected(monkeypatch) -> None:
    client = TestClient(app)

    def missing_product(_url: str, **_kwargs) -> ProductPageAnalysis:
        raise ProductNotDetectedError("No product signals found.")

    monkeypatch.setattr(product_routes, "analyze_product_url", missing_product)

    response = client.post("/api/v1/scan", json={"url": "https://example.com/nope"})

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "product_not_detected"


def test_scan_returns_product_page_unavailable(monkeypatch) -> None:
    client = TestClient(app)

    def unavailable(_url: str, **_kwargs) -> ProductPageAnalysis:
        raise PageFetchError("Request failed.")

    monkeypatch.setattr(product_routes, "analyze_product_url", unavailable)

    response = client.post("/api/v1/scan", json={"url": "https://example.com/product"})

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "product_page_unavailable"


def test_scan_validates_url_required() -> None:
    client = TestClient(app)

    response = client.post("/api/analyze-product", json={})

    assert response.status_code == 422


def test_scan_rejects_extra_request_fields() -> None:
    client = TestClient(app)

    response = client.post(
        "/api/v1/scan",
        json={
            "url": "https://example.com/product/123",
            "product_title": "Client supplied facts are not accepted",
        },
    )

    assert response.status_code == 422


def test_scan_calls_persistence_hook(monkeypatch) -> None:
    client = TestClient(app)
    calls = []
    monkeypatch.setattr(product_routes, "analyze_product_url", _analysis_for)

    def fake_save_scan(**kwargs) -> bool:
        calls.append(kwargs)
        return True

    monkeypatch.setattr(product_routes, "save_scan", fake_save_scan)

    response = client.post(
        "/api/v1/scan",
        json={"url": "https://example.com/product/123", "browser_id": "browser-1"},
    )

    assert response.status_code == 200
    assert calls
    assert calls[0]["browser_id"] == "browser-1"
    assert str(calls[0]["response"].scan_id) == response.json()["scan_id"]


def test_model_info_returns_runtime_configuration() -> None:
    client = TestClient(app)

    response = client.get("/api/v1/model-info")

    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == "0.3.0"
    assert "trustscore_weights" in body
    assert body["feedback_scoring"] == "not_applied"
    assert body["market_reference"]["provider"] == "serper"
    assert "api_key" not in body["market_reference"]
    assert body["fake_review_artifact_status"] in {"loaded", "missing_or_unavailable"}
    assert set(body["risk_model_artifact_status"]) == {"seller", "price", "policy"}


def test_feedback_endpoint_accepts_local_mode_feedback() -> None:
    client = TestClient(app)

    response = client.post(
        "/api/v1/feedback",
        json={
            "scan_id": "11111111-1111-4111-8111-111111111111",
            "helpful": True,
            "issue_category": "wrong_price",
            "corrected_component": "price_safety",
            "comment": "Useful",
        },
    )

    assert response.status_code == 200
    assert response.json() == {"status": "accepted"}


def test_feedback_rejects_invalid_scan_id() -> None:
    client = TestClient(app)

    response = client.post(
        "/api/v1/feedback",
        json={"scan_id": "scan-1", "helpful": True},
    )

    assert response.status_code == 422


def test_feedback_returns_not_found_for_unknown_scan(monkeypatch) -> None:
    client = TestClient(app)

    class Result:
        status = "missing"

    monkeypatch.setattr(product_routes, "save_feedback", lambda _payload: Result())

    response = client.post(
        "/api/v1/feedback",
        json={"scan_id": "11111111-1111-4111-8111-111111111111", "helpful": True},
    )

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "scan_not_found"
