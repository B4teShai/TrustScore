# Codex Prompts

Use these prompts one by one. Start with the master prompt, then continue with each implementation prompt.

## Prompt 1: Master implementation prompt

```text
You are implementing a project called AI TrustScore Browser Extension for Online Shopping.

Read the provided docs in the repository:
- README.md
- SYSTEM_ARCHITECTURE.md
- AI_ML_MODEL_SPEC.md
- DATASETS_AND_TRAINING.md
- API_SPEC.md
- DATABASE_SCHEMA.md
- EXTENSION_FRONTEND_SPEC.md
- ENVIRONMENT_AND_REQUIREMENTS.md
- IMPLEMENTATION_PLAN.md
- TESTING_AND_EVALUATION.md
- PROJECT_CONTEXT_FOR_CODEX.md

Build a clean MVP with this stack:
- Chrome Extension Manifest V3
- React + TypeScript + Vite for popup UI
- Python FastAPI backend
- Hugging Face Transformers for sentiment analysis
- scikit-learn Random Forest for fake review detection
- Supabase PostgreSQL schema and logging

Prioritize correctness and simplicity. Do not collect personal data. Do not make legal accusations about sellers. Use risk wording and explanations.

Start by creating the monorepo structure, backend health endpoint, extension skeleton, shared types, and a README with run instructions. Then wait for my review before implementing deeper ML features.
```

## Prompt 2: Create backend API skeleton

```text
Implement the FastAPI backend in apps/api.

Requirements:
1. Create app/main.py with FastAPI app.
2. Add CORS middleware.
3. Add GET /health.
4. Add POST /api/v1/scan with Pydantic request/response schemas from API_SPEC.md.
5. Add POST /api/v1/feedback.
6. Add GET /api/v1/model-info.
7. For now, return a mock TrustScore response from /scan.
8. Add requirements.txt and .env.example.
9. Add pytest tests for /health and /scan.

Use type hints and clean module structure.
```

## Prompt 3: Implement TrustScore Engine

```text
Implement the TrustScore Engine in the FastAPI backend.

Requirements:
1. Create services/trustscore_engine.py.
2. Implement weighted formula:
   - 0.30 review_authenticity
   - 0.20 seller_reliability
   - 0.20 sentiment
   - 0.15 return_policy_clarity
   - 0.10 price_safety
   - 0.05 user_feedback_history
3. Clamp all component scores to 0-100.
4. If a component is missing, use neutral score 50 and reduce confidence.
5. Implement risk classification:
   - >= 80 Low Risk
   - >= 50 Medium Risk
   - else High Risk
6. Implement top 3 explanation reasons based on weakest scores.
7. Add unit tests for formula, thresholds, missing data, and explanations.
```

## Prompt 4: Implement preprocessing and feature extraction

```text
Implement preprocessing and feature extraction for product reviews.

Requirements:
1. Create services/preprocessing.py.
2. Clean review text: lowercase, strip whitespace, remove HTML tags, remove empty reviews.
3. Detect duplicate reviews.
4. Compute review length statistics.
5. Compute short 5-star review rate.
6. Compute duplicate review percentage.
7. Compute negative keyword rate using keywords: fake, broken, scam, refund, late, poor quality, never arrived.
8. Compute simple rating-sentiment mismatch placeholder, to be improved after sentiment model exists.
9. Add tests.
```

## Prompt 5: Implement AI/ML inference services

```text
Implement AI/ML inference services.

Requirements:
1. Create services/sentiment_service.py using Hugging Face Transformers pipeline if available.
2. Use model name from env SENTIMENT_MODEL_NAME, default distilbert-base-uncased-finetuned-sst-2-english.
3. Return product-level sentiment_score 0-100.
4. Create services/fake_review_service.py.
5. If FAKE_REVIEW_MODEL_PATH and vectorizer exist, load joblib artifacts and predict fake_probability.
6. If artifacts do not exist, use a heuristic fallback based on duplicate rate, short 5-star rate, negative keywords, and mismatch rate.
7. Create services/risk_service.py for seller, price, and policy scores.
8. Integrate all services into /api/v1/scan.
9. Add tests for fallback behavior.
```

## Prompt 6: Implement database schema and logging

```text
Implement database support.

Requirements:
1. Add db/schema.sql using DATABASE_SCHEMA.md.
2. Add database connection using SQLAlchemy or Supabase client.
3. Add repository functions to upsert seller, insert product, insert reviews, insert prediction_run, insert model_predictions, insert user_feedback.
4. Integrate logging into /api/v1/scan and /api/v1/feedback.
5. If database is unavailable in local mode, log a warning and continue returning predictions.
6. Add tests using a mock repository.
```

## Prompt 7: Implement Chrome Extension and React popup

```text
Implement the Chrome Extension in apps/extension.

Requirements:
1. Use React + TypeScript + Vite.
2. Add Manifest V3 manifest.json.
3. Read the active-tab URL and a small visible product preview in the popup.
4. Send `{ url }` to `POST /api/v1/scan`; fall back to `/api/v1/scan-extracted` only when backend fetching is blocked; only send `browser_id` with feedback.
5. Add popup UI with these states:
   - loading
   - result
   - error
6. Show TrustScore, risk level, confidence, top 3 reasons, recommendation, and component scores.
7. Add Helpful and Not Helpful buttons that call /api/v1/feedback.
8. Store last result in chrome.storage.local.
9. Make the UI clean enough for a school presentation demo.
```

## Prompt 8: Implement fake review training script

```text
Implement a training script for the fake review model.

Requirements:
1. Create ml/training/train_fake_review_model.py.
2. Load the Hugging Face dataset theArijitDas/Fake-Reviews-Dataset.
3. Use text, rating, and category fields.
4. Build a scikit-learn pipeline with TF-IDF and RandomForestClassifier.
5. Split train/test.
6. Evaluate accuracy, precision, recall, F1, and confusion matrix.
7. Save model and vectorizer artifacts to ml/artifacts.
8. Save model_metadata.json with dataset name, version, metrics, and timestamp.
9. Make the script runnable from command line.
```

## Prompt 9: Add final tests and demo documentation

```text
Finish the MVP.

Requirements:
1. Add backend tests for endpoints, preprocessing, risk services, and TrustScore engine.
2. Add extension type checks and build verification.
3. Add sample payload JSON files.
4. Update README with setup steps:
   - backend setup
   - extension setup
   - database setup
   - model training
   - demo flow
5. Add troubleshooting section.
6. Ensure no real secrets are committed.
7. Ensure the project can be explained in a presentation.
```

## Prompt 10: Debugging prompt for Codex

```text
The implementation has errors. Please inspect the repository, run the tests/build commands, identify the root cause, and fix the smallest necessary set of files. Preserve the architecture from the docs. Do not rewrite unrelated parts. After fixing, summarize what changed and how to run the project.
```

## Prompt 11: Refactor prompt for Codex

```text
Refactor the project for readability and maintainability.

Rules:
1. Keep the public API contract unchanged.
2. Keep the TrustScore formula unchanged.
3. Keep extension UI behavior unchanged.
4. Improve module names, type hints, error handling, and tests.
5. Do not add unnecessary dependencies.
6. Summarize changes after refactor.
```
