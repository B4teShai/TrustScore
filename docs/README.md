# AI TrustScore Browser Extension

AI TrustScore is a prototype Chrome Extension and FastAPI backend for analyzing online shopping product pages. The extension previews visible active-tab product fields, sends the URL to the backend first, and falls back to a minimal active-tab payload only when the retailer blocks backend fetching.

The current MVP includes a health check, product-analysis inference endpoints, backend product-page extraction, active-tab fallback scoring, model artifact loading with deterministic fallbacks, optional database persistence behind `DATABASE_URL`, and a popup workflow that renders product previews plus TrustScore results.

## Project Structure

```text
apps/
  api/            FastAPI backend
  extension/      Chrome Extension frontend built with React, TypeScript, and Vite
db/               Supabase/PostgreSQL schema and future migrations
ml/               AI/ML training code, notebooks, data, and model artifacts
```

## Backend: Local Run

```bash
cd apps/api
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8000
```

Then open:

```text
http://localhost:8000/health
```

Expected response:

```json
{
  "status": "ok",
  "service": "ai-trustscore-api"
}
```

Try the canonical product-analysis endpoint:

```bash
curl -X POST http://localhost:8000/api/v1/scan \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/product/123"
  }'
```

The response is a deterministic inference MVP result using preprocessing, rule-based risk scoring, sentiment fallback, and fake-review fallback when trained artifacts are missing. Example:

```json
{
  "scan_id": "7f7e2a3c-4f0b-4b1c-9a9d-1f6c1d76eae7",
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
  "confidence": 0.58,
  "component_scores": {
    "review_authenticity": 69,
    "seller_reliability": 73,
    "sentiment": 65,
    "return_policy_clarity": 80,
    "price_safety": 50,
    "user_feedback_history": 50
  },
  "top_reasons": [
    "Price should be compared with similar products.",
    "Review sentiment is mixed."
  ],
  "evidence": [
    {
      "component": "price_safety",
      "summary": "Price safety is scored only with a verified same-currency market reference.",
      "evidence": [
        "Listed price only: USD 29.99",
        "No verified market reference found."
      ],
      "missing_inputs": [],
      "confidence": 0
    }
  ],
  "missing_inputs": [],
  "score_semantics": "TrustScore is normalized over non-feedback evidence.",
  "recommendation": "Check seller details, recent reviews, and return policy before buying.",
  "model_version": "0.3.0",
  "fetch_mode": "http",
  "extraction_signals": ["structured_product", "price", "seller"],
  "model_modes": {
    "fake_review": "heuristic_fallback",
    "sentiment": "keyword_fallback",
    "seller_reliability": "rule_fallback",
    "price_safety": "rule_fallback",
    "return_policy_clarity": "rule_fallback",
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

Run backend and ML tests:

```bash
PYTHONPATH=apps/api pytest apps/api/tests -q
PYTHONPATH=. pytest ml/tests -q
```

## Extension: Local Build

```bash
cd apps/extension
npm ci
npm run lint
npm test -- --run
npm run build
```

Load the extension in Chrome:

1. Open `chrome://extensions`.
2. Enable Developer mode.
3. Click Load unpacked.
4. Select `apps/extension/dist`.

For local API calls, the extension defaults to:

```text
http://localhost:8000
```

You can override it with `VITE_API_BASE_URL` before building.

For workflow testing:

1. Start the backend at `http://localhost:8000`.
2. Open a real HTTP/HTTPS shopping product-detail page.
3. Open the extension popup.
4. Confirm the popup previews product name, image, seller, and visible price/review signals.
5. Click `Analyze product`.
6. Confirm the TrustScore result appears.

The popup sends the active-tab URL to `/api/v1/scan` first. If the retailer blocks backend fetching, it sends the small active-tab preview to `/api/v1/scan-extracted`. It stores the last result in `chrome.storage.local`; a browser ID is created only when feedback is submitted.

## Database Setup

Option A: Supabase

1. Create a Supabase project.
2. Open the SQL Editor.
3. Run `db/schema.sql`, or apply the SQL files in `db/migrations` in order.
4. Copy the Supabase database URL into `apps/api/.env`.
5. Set a production `BROWSER_ID_HASH_SALT`.

Option B: Local PostgreSQL

1. Create a database named `ai_trustscore`.
2. Run `db/schema.sql`, or apply the SQL files in `db/migrations` in order.
3. Set `DATABASE_URL` in `apps/api/.env`.

Never commit real Supabase keys or database passwords.

## Created Files

### Root

- `.env.example`: shared local defaults, including the backend base URL.
- `.gitignore`: ignores local env files, Python caches, virtualenvs, Node dependencies, and build output.
- `README.md`: this run guide and scaffold explanation.

### Backend

- `apps/api/.env.example`: backend environment template for CORS, database, Supabase, model names, artifact paths, and scoring weights.
- `apps/api/requirements.txt`: Python dependencies planned for FastAPI, ML, database access, and tests.
- `apps/api/app/main.py`: FastAPI application entrypoint with CORS, `GET /health`, and API router registration.
- `apps/api/app/api/product_analysis.py`: route module for `POST /api/analyze-product`, `POST /api/v1/scan`, `POST /api/v1/feedback`, and `GET /api/v1/model-info`.
- `apps/api/app/core/config.py`: small environment loader used by the backend.
- `apps/api/app/schemas/product_analysis.py`: Pydantic request and response models for product analysis.
- `apps/api/app/page_fetching/`: bounded public web-page retrieval with HTTP-first fetching and opt-in rendered fallback.
- `apps/api/app/extraction/`: backend product-page parser for product metadata, price, seller, reviews, and policy text.
- `apps/api/app/services/product_scoring.py`: product-level inference orchestration for the TrustScore response.
- `apps/api/app/services/trustscore_engine.py`: weighted TrustScore formula, risk classification, recommendations, and explanation reasons.
- `apps/api/app/services/mock_scoring.py`: legacy mock service kept for compatibility, no longer used by the API route.
- `apps/api/app/__init__.py`: marks the backend app as a Python package.
- `apps/api/app/api/__init__.py`: package placeholder for future API route modules.
- `apps/api/app/core/__init__.py`: package placeholder for shared settings and utilities.
- `apps/api/app/db/`: lazy SQLAlchemy session and repository functions for optional scan and feedback persistence.
- `apps/api/app/ml/`: AI/ML inference services for preprocessing, sentiment fallback, fake-review fallback/artifact loading, and rule-based seller/price/policy scoring.
- `apps/api/app/models/__init__.py`: package placeholder for database model definitions.
- `apps/api/app/schemas/__init__.py`: package placeholder for Pydantic request and response schemas.
- `apps/api/app/services/__init__.py`: package placeholder for business logic services.
- `apps/api/tests/test_health.py`: smoke test that verifies the health endpoint response.
- `apps/api/tests/test_product_analysis.py`: tests for product-analysis endpoints, model info, feedback, and validation.
- `apps/api/tests/test_ml_inference.py`: tests for preprocessing, fake-review fallback, risk scoring, and TrustScore formula behavior.

### Extension

- `apps/extension/package.json`: npm scripts and dependencies for the React + TypeScript + Vite extension.
- `apps/extension/package-lock.json`: locked npm dependency versions generated by `npm install`.
- `apps/extension/tsconfig.json`: TypeScript compiler settings for strict extension code.
- `apps/extension/vite.config.ts`: Vite build config for popup, background worker, and manifest copying.
- `apps/extension/index.html`: popup HTML entry loaded by Chrome.
- `apps/extension/manifest.json`: Chrome Manifest V3 configuration with active-tab, scripting, storage, and local API host permissions.
- `apps/extension/public/icons/.gitkeep`: keeps the icons folder available for future extension icons.
- `apps/extension/src/vite-env.d.ts`: Vite TypeScript environment reference.
- `apps/extension/src/background/background.ts`: minimal background service worker for install lifecycle logging.
- `apps/extension/src/popup/main.tsx`: React popup bootstrap file.
- `apps/extension/src/popup/App.tsx`: popup workflow that previews active-tab product fields, calls the backend, and displays extracted product metadata plus TrustScore results.
- `apps/extension/src/popup/App.css`: popup styling with risk colors and component score bars.
- `apps/extension/src/shared/types.ts`: shared TypeScript types matching the planned API payloads and responses.
- `apps/extension/src/shared/apiClient.ts`: typed API helper for health, canonical scan, active-tab fallback scan, feedback calls, structured error parsing, and response validation.

### Database

- `db/schema.sql`: Supabase/PostgreSQL schema for sellers, products, reviews, model versions, prediction runs, model predictions, and hashed user feedback.
- `db/migrations/`: versioned SQL bootstrap and hardening migrations.

### ML

- `ml/README.md`: explains the purpose of the ML folders.
- `ml/training/download_datasets.py`: downloads sampled reproducible datasets into `ml/data`.
- `ml/training/train_all.py`: runs fake-review, sentiment, risk, and optional calibration training.
- `ml/notebooks/.gitkeep`: keeps the future notebook folder.
- `ml/artifacts/.gitkeep`: keeps the future saved model artifact folder.
- `ml/data/.gitkeep`: keeps the future local data folder.

## Next Step

Recommended next backend step:

1. Keep `ENABLE_RENDERED_FETCH=0` for local extension demos; set it to `1` and install Playwright browsers only if backend-rendered product pages are required.
2. Run `python ml/training/download_datasets.py` to populate `ml/data`.
3. Run `python ml/training/train_all.py --skip-download` after the sampled data is available.
4. Add human-labeled TrustScore calibration data before making production fraud claims.
