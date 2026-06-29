# Implementation Plan for Codex

## Goal
Implement a working MVP of AI TrustScore Browser Extension with backend AI/ML inference and feedback logging.

## Milestone 1: Repository setup

Tasks:
1. Create monorepo structure.
2. Add backend FastAPI app skeleton.
3. Add extension React + TypeScript + Vite skeleton.
4. Add shared API types.
5. Add README run instructions.

Acceptance criteria:
- `apps/api` starts with `uvicorn app.main:app --reload`.
- `apps/extension` builds successfully.
- `GET /health` returns status ok.

## Milestone 2: Backend scan API

Tasks:
1. Implement Pydantic schemas.
2. Implement `/api/v1/scan` endpoint.
3. Implement `/api/v1/feedback` endpoint.
4. Add CORS middleware.
5. Return mock TrustScore first.

Acceptance criteria:
- Can send sample JSON to `/api/v1/scan` and get response.
- Can submit feedback.

## Milestone 3: TrustScore Engine

Tasks:
1. Implement scoring config.
2. Implement weighted TrustScore formula.
3. Implement risk classification.
4. Implement top reason selection.
5. Add unit tests.

Acceptance criteria:
- Scores match expected formula.
- Risk classification is correct:
  - 80 to 100: Low Risk.
  - 50 to 79: Medium Risk.
  - 0 to 49: High Risk.

## Milestone 4: AI/ML inference services

Tasks:
1. Implement sentiment service using Hugging Face Transformers.
2. Implement fake review service with scikit-learn Random Forest if artifacts exist.
3. Add fallback fake review heuristic if artifacts are missing.
4. Implement seller, price, and policy scoring.
5. Combine all outputs into component scores.

Acceptance criteria:
- Backend returns component scores.
- Backend handles missing fields without crashing.
- Backend returns confidence score.

## Milestone 5: Database integration

Tasks:
1. Add SQL schema.
2. Add database connection.
3. Store sellers, products, reviews, prediction runs, model predictions, feedback.
4. Add repository/service layer.

Acceptance criteria:
- Scan results are saved.
- Feedback is saved.
- API still works if database is disabled in local mode, with warning.

## Milestone 6: Extension product preview workflow

Tasks:
1. Read the active tab URL from the popup.
2. Read a small active-tab product preview using `activeTab` and `scripting`.
2. Send `{ url }` to `POST /api/v1/scan`.
3. Fall back to `POST /api/v1/scan-extracted` when the retailer blocks backend fetching.
4. Render the TrustScore response.
5. Store the last result in Chrome storage.
6. Submit feedback with a hashed-on-backend browser ID.

Acceptance criteria:
- Open product page, click extension, see title, image, seller, and analyze button.
- Missing fields are handled gracefully.

## Milestone 7: Popup UI

Tasks:
1. Build popup components.
2. Add score badge and risk color.
3. Show top 3 reasons.
4. Add helpful/not helpful feedback buttons.
5. Add error and loading states.

Acceptance criteria:
- Popup looks presentable for PPT demo.
- Feedback submission works.

## Milestone 8: Model training script

Tasks:
1. Add training script for Fake Reviews Dataset.
2. Load dataset from Hugging Face datasets library.
3. Train TF-IDF + RandomForest pipeline.
4. Evaluate accuracy, precision, recall, F1.
5. Save artifacts to `ml/artifacts`.

Acceptance criteria:
- Training script runs on sample data.
- Artifacts are saved.
- Metrics are printed and written to metadata JSON.

## Milestone 9: Tests and final polish

Tasks:
1. Add backend unit tests.
2. Add basic extension tests or type checks.
3. Add sample input JSON.
4. Add screenshots or demo instructions.
5. Update README.

Acceptance criteria:
- `pytest` passes.
- `npm run build` passes.
- Demo steps are documented.

## Suggested MVP demo script

1. Start FastAPI backend.
2. Load Chrome extension locally.
3. Open an example product page.
4. Click extension popup.
5. Show TrustScore, risk level, and top reasons.
6. Click Helpful or Not Helpful.
7. Show feedback saved in backend/database logs.

## Implementation priorities
If time is limited:
1. Backend API and scoring engine.
2. Popup UI.
3. Basic content extraction.
4. Heuristic fake review scoring.
5. Database and training script.
6. Full ML artifact integration.
