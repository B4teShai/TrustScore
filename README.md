# AI TrustScore

A Chrome extension + FastAPI backend that analyses an online product page and returns an
explainable **trust score (0–100)**, a **risk level** (Low / Medium / High) and short reasons.
It combines five ML signals — fake-review authenticity, sentiment, seller reliability, price
safety and return-policy clarity — into one weighted score.

```
TrustScore/
├── apps/
│   ├── api/         FastAPI backend (serves the models)
│   └── extension/   Chrome extension (React + TypeScript, Manifest V3)
├── ml/              training pipelines + datasets + model artifacts (v1 / v2 / v3)
└── presentation/    slides & reports
```

The models are **already trained and committed** under `ml/artifacts/`, so you can run the
system without training anything. Training is optional — see [ml/README.md](ml/README.md).

### Model artifacts: best + version by version

- `ml/artifacts/best/` — the curated **best** model per signal (one directory, one
  `best_manifest.json`). The Docker stack and `TRUSTSCORE_MODEL_VERSION=best` use this set.
- `ml/artifacts/{v1,v2,v3}/` — every trained version is kept **version by version** for
  comparison and reproducibility.

| Signal | Best model | From |
|---|---|---|
| Review authenticity | Calibrated LinearSVC + word/char TF-IDF | v3 |
| Sentiment | TF-IDF + LogReg (neg/neu/pos) | v2 |
| Price safety | IsolationForest anomaly + ratio rule | v3 |
| Seller reliability | Transparent rule | v3 |
| Return-policy clarity | Transparent rule | v3 |

Rebuild the best set anytime: `python -m ml.training.make_best_artifacts`.

---

## Run everything with Docker (one command)

Starts Postgres, the API (on the **best** model set, persisting to Postgres), the extension
dev server, and a trainer step that assembles `ml/artifacts/best/`.

```bash
cp .env.docker.example .env     # optional: paste ANTHROPIC_API_KEY for Claude feedback
docker compose up --build
```

| Service | URL / port | Notes |
|---|---|---|
| API | http://localhost:8000 | `/health`, `/api/v1/model-info` |
| Extension (Vite) | http://localhost:5173 | dev server with hot reload |
| Postgres | localhost:5432 | db `trustscore`, `postgres` / `postgres`, migrations auto-applied |

To use the extension **in Chrome**, build `dist/` and load it unpacked (see step 2 below);
Chrome can't run inside the container. Run real training on demand:

```bash
docker compose run --rm trainer python -m ml.training.train_all
```

The rest of this README covers running the same pieces **without** Docker.

---

## Prerequisites

- **Python 3.11+** (developed on 3.13)
- **Node.js 18+** (developed on 25) and **npm**
- **Google Chrome** (or any Chromium browser)

---

## 1. Start the backend (API)

The extension expects the API at **http://localhost:8000**.

```bash
cd apps/api

# create + activate a virtual environment (first time only)
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# install dependencies (first time only)
pip install -r requirements.txt

# run the API on port 8000
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Check it's up:

```bash
curl http://localhost:8000/health           # -> {"status":"ok","service":"ai-trustscore-api"}
curl http://localhost:8000/api/v1/model-info # shows model version + AI-feedback + persistence
```

### Claude AI feedback (paste your API key here)

The TrustScore and component scores are always produced by the committed v3 ML
models. The short **"Shopping Guidance"** shown to the shopper can additionally be
written by **Claude** for a clearer, evidence-grounded explanation.

1. Create your env file (this is the file you paste the key into):
   ```bash
   cd apps/api
   cp .env.example .env
   ```
2. Open **`apps/api/.env`** and paste your key on the `ANTHROPIC_API_KEY=` line:
   ```env
   ANTHROPIC_API_KEY=sk-ant-...     # from https://console.anthropic.com
   ANTHROPIC_MODEL=claude-haiku-4-5
   ```
3. Restart the API. `/api/v1/model-info` will then show `"ai_feedback": {"active": true, ...}`
   and each result's `recommendation_source` will be `"ai"` (the popup shows a **Claude** badge).

Leave `ANTHROPIC_API_KEY` blank to use the built-in rule-based guidance — the score
is identical either way; only the wording of the recommendation changes.

**Language & currency.** The extension detects the product page's language (`<html lang>`,
`content-language`, or `og:locale`) and sends it as `locale` — primary targets are
**Japanese (`ja`) and English (`en`)**. With a key set, Claude writes the recommendation
**and** translates the reasons into that language in one call; the response echoes `language`.
Prices are shown in the page's own detected currency (no conversion), with correct formatting
per currency — e.g. **yen** (`¥`/`円`/`JPY`) renders as `￥2,980` (no decimals), USD as
`$29.99`. Without a key, guidance/reasons stay English. (Static popup labels like "Top reasons"
remain English for now.)

### Saving scored pages

Every scored page is persisted. With `DATABASE_URL` set it goes to Postgres
(`db/migrations/`). With no database, it is appended to **`apps/api/.data/scans.jsonl`**
(one JSON record per scan, including the AI feedback used) so demos keep a durable
record. Set `PERSIST_LOCAL_SCANS=0` to disable the local store.

### Choosing a model version (optional)

By default the API serves the **v3 production model set**: calibrated TF-IDF fake-review
detection, v2 sentiment, transparent seller/policy rules, and leakage-free price anomaly
semantics. The **best** set (used by Docker) is the same winners gathered into
`ml/artifacts/best/`:

```bash
TRUSTSCORE_MODEL_VERSION=best uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
# /api/v1/model-info reports model_version 1.0.0

TRUSTSCORE_MODEL_VERSION=v2 uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
# /api/v1/model-info will then report model_version 0.2.0

TRUSTSCORE_MODEL_VERSION=v1 uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
# /api/v1/model-info will then report model_version 0.1.0
```

> A repo already exists at the project root with a shared `.venv`. If you prefer, you can reuse
> it instead of creating one inside `apps/api`:
> `pip install -r apps/api/requirements.txt` then run uvicorn from `apps/api`.

---

## 2. Start the extension (frontend)

In a **second terminal**:

```bash
cd apps/extension

npm install        # first time only
npm run build      # compiles to apps/extension/dist/
```

Load it into Chrome:

1. Open `chrome://extensions`.
2. Turn on **Developer mode** (top-right).
3. Click **Load unpacked** and select the **`apps/extension/dist`** folder.
4. Pin the **AI TrustScore** extension from the toolbar puzzle icon.

### Use it

1. Make sure the API (step 1) is running on port 8000.
2. Open any product page (e.g. an Amazon product).
3. Click the **AI TrustScore** icon → **Analyze product**.
4. You'll see the trust score, risk level, confidence, missing evidence, component scores,
   and evidence used.

To iterate on the UI with hot-reload instead of rebuilding: `npm run dev` (Vite dev server).

---

## Quick end-to-end test (no extension needed)

With the API running, you can score a product directly:

```bash
curl -s -X POST http://localhost:8000/api/v1/scan-extracted \
  -H "Content-Type: application/json" \
  -d '{"product":{"url":"https://example.com/p","product_title":"Wireless Earbuds",
       "price":29.99,"average_market_price":59.99,
       "seller":{"name":"Example Store","rating":4.6,"review_count":1200},
       "return_policy":"30-day return and refund with warranty",
       "reviews":[{"text":"Great sound and battery life","rating":5}]}}'
```

You'll get back `trust_score`, `risk_level`, `confidence`, `component_scores`, `top_reasons`,
`evidence`, `missing_inputs`, `score_semantics`, and model provenance.

---

## Running the tests

```bash
# backend
cd apps/api && pytest

# ML pipeline
cd <repo-root> && PYTHONPATH=. python -m pytest ml/tests

# extension
cd apps/extension && npm test
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Extension shows a connection error | The API isn't running, or not on port 8000. Start it (step 1). |
| `CORS` / blocked request | Start the API with `CORS_ALLOW_ORIGINS` including your extension origin, or keep the default which allows the extension. |
| `/api/v1/model-info` shows `missing_or_unavailable` | The model artifacts aren't found. Confirm `ml/artifacts/` exists; for v2 set `TRUSTSCORE_MODEL_VERSION=v2`. |
| Port 8000 already in use | Run on another port and set `VITE_API_BASE_URL` for the extension build accordingly, or free the port. |
| `playwright` errors on rendered fetch | Rendered fetch is **off by default**. Leave `ENABLE_RENDERED_FETCH=0` (no browser install needed). |

---

## More

- ML training, versions and results: [ml/README.md](ml/README.md)
- v3 research findings: `ml/reports/v3/FINDINGS.md` and `ml/reports/v3/RESULTS.md`
- Presentation: `presentation/presentation-2/`
