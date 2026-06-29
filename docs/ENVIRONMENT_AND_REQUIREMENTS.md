# Environment and Requirements

## Development environment

Recommended local environment:

| Tool | Recommended version |
|---|---|
| Node.js | 20 LTS or newer |
| Python | 3.11 or newer |
| PostgreSQL | Supabase hosted Postgres or local Postgres 15+ |
| Browser | Google Chrome or Chromium |
| Package manager | npm, using `package-lock.json` and `npm ci` |

## Backend Python requirements
Create `apps/api/requirements.txt`:

```txt
fastapi
uvicorn[standard]
pydantic
pydantic-settings
python-dotenv
httpx
numpy
pandas
scikit-learn
joblib
transformers
torch
SQLAlchemy
psycopg[binary]
supabase
pytest
pytest-asyncio
```

For a smaller MVP, `supabase` can be skipped if using SQLAlchemy with Postgres directly.

## Extension package.json example
`apps/extension/package.json` pins dependency versions so `npm ci` is repeatable:

```json
{
  "name": "ai-trustscore-extension",
  "version": "0.1.0",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview",
    "lint": "eslint .",
    "test": "vitest"
  },
  "dependencies": {
    "react": "19.2.6",
    "react-dom": "19.2.6"
  },
  "devDependencies": {
    "@types/chrome": "0.1.42",
    "@types/react": "19.2.14",
    "@types/react-dom": "19.2.3",
    "@vitejs/plugin-react": "6.0.1",
    "typescript": "6.0.3",
    "vite": "8.0.11",
    "vitest": "4.1.5"
  }
}
```

## Backend environment variables
Create `apps/api/.env.example`:

```env
APP_NAME=AI TrustScore API
APP_ENV=local
API_HOST=0.0.0.0
API_PORT=8000
CORS_ALLOW_ORIGINS=http://localhost:5173,chrome-extension://YOUR_EXTENSION_ID

DATABASE_URL=postgresql+psycopg://postgres:password@localhost:5432/ai_trustscore
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=replace-with-local-secret

MODEL_VERSION=0.1.0
BROWSER_ID_HASH_SALT=replace-with-random-production-salt
ENABLE_RENDERED_FETCH=0
SENTIMENT_MODEL_NAME=distilbert-base-uncased-finetuned-sst-2-english
FAKE_REVIEW_MODEL_PATH=../../ml/artifacts/fake_review_rf.joblib
FAKE_REVIEW_VECTORIZER_PATH=../../ml/artifacts/fake_review_vectorizer.joblib

TRUST_WEIGHT_REVIEW_AUTHENTICITY=0.30
TRUST_WEIGHT_SELLER_RELIABILITY=0.20
TRUST_WEIGHT_SENTIMENT=0.20
TRUST_WEIGHT_POLICY=0.15
TRUST_WEIGHT_PRICE=0.10
TRUST_WEIGHT_FEEDBACK=0.05
```

Never commit real Supabase keys.

## Backend local setup

```bash
cd apps/api
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8000
```

On Windows PowerShell:

```powershell
cd apps/api
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --reload --port 8000
```

## Extension local setup

```bash
cd apps/extension
npm ci
npm test -- --run
npm run build
```

Then load the built extension in Chrome:

1. Open `chrome://extensions`.
2. Enable Developer mode.
3. Click Load unpacked.
4. Select the build output folder.

## Database setup

Option A: Supabase
1. Create a Supabase project.
2. Open SQL Editor.
3. Run `db/schema.sql`, or run the versioned migrations in `db/migrations/` in order.
4. Copy database connection string into `.env`.

Option B: Local Postgres
1. Create database `ai_trustscore`.
2. Run `db/schema.sql`, or run the versioned migrations in `db/migrations/` in order.
3. Use local `DATABASE_URL`.

## Model artifacts
For MVP, if model artifacts are missing:
- Use pre-trained Hugging Face sentiment pipeline.
- Use fallback rule-based fake-review scoring.
- Log warning: `fake_review_model_missing`.

Expected model artifacts:

```text
ml/artifacts/fake_review_rf.joblib
ml/artifacts/fake_review_vectorizer.joblib
ml/artifacts/model_metadata.json
```

## Deployment notes
- Backend can be deployed to Render, Fly.io, Railway, or any container host.
- Supabase remains managed database.
- Extension must call deployed backend URL.
- For Chrome Web Store release, reduce permissions and prepare privacy policy.
