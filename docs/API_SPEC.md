# Backend API Specification

## Base URL
Local development:

```text
http://localhost:8000
```

Production example:

```text
https://api.example.com
```

## Authentication for MVP
No user login is required for MVP. The popup sends the active-tab URL for the primary scan, can send a small active-tab preview to the fallback scan endpoint, and creates an optional anonymous browser ID only when feedback is submitted.

## Endpoints

### 1. Health check

```http
GET /health
```

Response:

```json
{
  "status": "ok",
  "service": "ai-trustscore-api"
}
```

### 2. Scan product page

```http
POST /api/v1/scan
Content-Type: application/json
```

Request body:

```json
{
  "url": "https://example.com/product/123"
}
```

Response body:

```json
{
  "scan_id": "uuid",
  "product": {
    "url": "https://example.com/product/123",
    "site": "example.com",
    "product_title": "Wireless Headphones",
    "product_image_url": "https://example.com/product-image.jpg",
    "price": 29.99,
    "currency": "USD",
    "seller_name": "Example Store"
  },
  "trust_score": 68,
  "risk_level": "Medium Risk",
  "confidence": 0.76,
  "component_scores": {
    "review_authenticity": 58,
    "seller_reliability": 80,
    "sentiment": 65,
    "return_policy_clarity": 45,
    "price_safety": 78,
    "user_feedback_history": 70
  },
  "top_reasons": [
    "Some reviews look repeated or suspicious.",
    "Return policy is unclear.",
    "Several reviews mention product quality problems."
  ],
  "evidence": [
    {
      "component": "price_safety",
      "summary": "Price safety needs both listed price and a market reference to flag anomalies.",
      "evidence": ["Listed price: USD 29.99", "Market reference: USD 59.99"],
      "missing_inputs": [],
      "confidence": 1.0
    }
  ],
  "missing_inputs": ["seller tenure"],
  "score_semantics": "TrustScore is normalized over non-feedback evidence. User feedback is stored for evaluation and is not used in the score until reviewed aggregates exist.",
  "recommendation": "Check return policy and seller details before buying.",
  "model_version": "0.3.0",
  "fetch_mode": "http",
  "extraction_signals": ["structured_product", "price", "seller"],
  "model_modes": {
    "fake_review": "calibrated_tfidf_artifact",
    "sentiment": "keyword_fallback",
    "seller_reliability": "transparent_rule_v3",
    "price_safety": "v3_price_ratio_anomaly",
    "return_policy_clarity": "transparent_rule_v3",
    "user_feedback_history": "not_applied"
  },
  "model_artifact_status": {
    "fake_review": "missing_or_unavailable",
    "sentiment": "missing_or_unavailable",
    "risk": {
      "seller": "missing_or_unavailable",
      "price": "missing_or_unavailable",
      "policy": "missing_or_unavailable"
    }
  },
  "model_versions": {
    "trustscore": "0.3.0",
    "fake_review": "0.3.0",
    "sentiment": "0.2.0",
    "risk": "0.3.0"
  },
  "is_mock": false
}
```

`POST /api/analyze-product` remains as a deprecated compatibility alias for the same URL-only request.

### 3. Scan active-tab extracted product fallback

`POST /api/v1/scan-extracted` is used by the extension only when `/api/v1/scan`
cannot fetch readable product-page HTML from the retailer. It accepts a minimal
active-tab preview, not raw page HTML.

```json
{
  "product": {
    "url": "https://example.com/product/123",
    "site": "example.com",
    "product_title": "Wireless Headphones",
    "product_image_url": "https://example.com/product-image.jpg",
    "price": 29.99,
    "currency": "USD",
    "seller": { "name": "Example Store" },
    "reviews": [],
    "rating": 4.3,
    "review_count": 120
  }
}
```

Successful responses use the same TrustScore response body as `/api/v1/scan`
with `fetch_mode: "extension_dom"`.

### 4. Submit feedback

```http
POST /api/v1/feedback
Content-Type: application/json
```

Request body:

```json
{
  "scan_id": "uuid",
  "browser_id": "anonymous-browser-id",
  "helpful": true,
  "issue_category": "wrong_price",
  "corrected_component": "price_safety",
  "expected_risk_level": "Medium Risk",
  "comment": "The warning was useful."
}
```

Response:

```json
{
  "status": "saved"
}
```

When `DATABASE_URL` is not configured, local demo mode returns:

```json
{
  "status": "accepted"
}
```

### 5. Get model info

```http
GET /api/v1/model-info
```

Response:

```json
{
  "model_version": "0.3.0",
  "model_version_tag": "v3",
  "score_semantics": "TrustScore is normalized over active non-feedback weights. Feedback is collected for evaluation and not used until reviewed aggregates exist.",
  "sentiment_model": "distilbert-base-uncased-finetuned-sst-2-english",
  "sentiment_artifact_status": "loaded",
  "fake_review_model": "v3_calibrated_tfidf_if_artifacts_exist_else_heuristic",
  "fake_review_artifact_status": "loaded",
  "risk_model_artifact_status": {
    "seller": "transparent_rule_v3",
    "price": "v3_anomaly_loaded",
    "policy": "transparent_rule_v3"
  },
  "trustscore_weights": {
    "review_authenticity": 0.30,
    "seller_reliability": 0.20,
    "sentiment": 0.20,
    "return_policy_clarity": 0.15,
    "price_safety": 0.10,
    "user_feedback_history": 0.0
  },
  "feedback_scoring": "not_applied"
}
```

## Pydantic schema suggestions

### ProductScanRequest

Fields:
- `url: str`
- `browser_id: str | None`

### SellerInfo
Fields:
- `name: str | None`
- `rating: float | None`
- `review_count: int | None`
- `years_active: int | None`

### ReviewInput
Fields:
- `text: str`
- `rating: int | float | None`
- `date: str | None`
- `verified_purchase: bool | None`

### ProductScanResponse
Fields:
- `scan_id: str`
- `product: ProductMetadata`
- `trust_score: int`
- `risk_level: "Low Risk" | "Medium Risk" | "High Risk"`
- `confidence: float`
- `component_scores: ComponentScores`
- `top_reasons: list[str]`
- `recommendation: str`
- `model_version: str`
- `fetch_mode: str`
- `extraction_signals: list[str]`
- `model_modes: dict[str, str]`
- `model_artifact_status: dict`
- `model_versions: dict[str, str]`

## Error handling

Return `400` when:
- Payload is too large.

Return `422` when:
- Pydantic validation fails.
- URL is not a supported public HTTP/HTTPS product page.
- Backend extraction cannot find enough product-detail evidence.
- The URL uses non-default HTTP/HTTPS ports, embedded credentials, local/private/link-local targets, unsupported content types, unsafe redirects, or an oversized response.

Return `404` when:
- Feedback references an unknown persisted scan ID.

Return `503` when:
- Feedback persistence is configured but temporarily unavailable.

Return `500` only for unexpected server errors.

## CORS
During local development, allow the extension origin or use permissive CORS for localhost only.

Example:

```python
allow_origins = [
    "chrome-extension://<extension-id>",
    "http://localhost:5173"
]
```

## Rate limit recommendation
For MVP, add simple rate limiting later:
- Max 30 scans per anonymous browser ID per hour.
- Max 10 feedback submissions per scan.
