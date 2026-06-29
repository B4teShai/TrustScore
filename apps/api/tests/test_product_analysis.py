from fastapi.testclient import TestClient

from app.api import product_analysis as product_routes
from app.main import app
from app.page_fetching.fetcher import PageFetchError
from app.schemas.product_analysis import ProductPageData, ReviewInput, SellerInfo
from app.services.product_page_analysis import ProductPageAnalysis, ProductNotDetectedError


def _expected_risk_level(score: int) -> str:
    if score >= 80:
        return "Low Risk"
    if score >= 50:
        return "Medium Risk"
    return "High Risk"


def _analysis_for(url: str) -> ProductPageAnalysis:
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
    assert len(body["product"]["currency"]) == 16
    assert len(body["product"]["seller_name"]) == 160


def test_scan_returns_product_not_detected(monkeypatch) -> None:
    client = TestClient(app)

    def missing_product(_url: str) -> ProductPageAnalysis:
        raise ProductNotDetectedError("No product signals found.")

    monkeypatch.setattr(product_routes, "analyze_product_url", missing_product)

    response = client.post("/api/v1/scan", json={"url": "https://example.com/nope"})

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "product_not_detected"


def test_scan_returns_product_page_unavailable(monkeypatch) -> None:
    client = TestClient(app)

    def unavailable(_url: str) -> ProductPageAnalysis:
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
