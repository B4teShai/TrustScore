# Project Context for Codex

Use this document as the first context file when asking Codex to implement the project.

## Project
AI TrustScore Browser Extension for Online Shopping.

## Goal
Build a working prototype that analyzes product-page data and gives:
- TrustScore from 0 to 100.
- Risk level: Low Risk, Medium Risk, or High Risk.
- Top 3 reasons.
- Shopping recommendation.
- Helpful/not helpful feedback option.

## Tech stack
- Extension: Chrome Extension Manifest V3.
- Frontend: React + TypeScript + Vite.
- Backend: Python FastAPI.
- AI/NLP: Hugging Face Transformers, BERT or DistilBERT.
- ML: scikit-learn Random Forest Classifier.
- Database: Supabase PostgreSQL.

## Main system workflow

```text
Product Page Data
-> Preprocessing
-> Feature Extraction
-> AI/ML Models
-> TrustScore Engine
-> Risk Level + Explanation
-> User Feedback
-> Feedback Storage
-> Offline Model Retraining
```

## Three model groups

### 1. Review Sentiment Model
Algorithm: BERT or DistilBERT.
Input: review text.
Output: sentiment score from 0 to 100.

### 2. Fake Review Detection Model
Algorithm: Random Forest.
Input features:
- repeated review text.
- short 5-star reviews.
- rating and sentiment mismatch.
- duplicate review percentage.
- average review length.
- review similarity.
Output: fake review probability and review authenticity score.

### 3. Seller, Price, and Policy Risk Model
Algorithm: rule-based scoring plus optional ML.
Input:
- seller rating.
- seller review count.
- product price compared to average market price.
- return policy clarity.
Output:
- seller reliability score.
- price safety score.
- return policy clarity score.

## TrustScore formula

```text
TrustScore =
      30% Review Authenticity
    + 20% Seller Reliability
    + 20% Sentiment Score
    + 15% Return Policy Clarity
    + 10% Price Safety
    + 5% User Feedback History
```

## Risk classification

```text
If TrustScore >= 80:
    Risk Level = Low Risk
Else if TrustScore >= 50:
    Risk Level = Medium Risk
Else:
    Risk Level = High Risk
```

## Explanation generation
Find the lowest component scores and convert them into human-readable reasons. Return top 3 reasons.

Example reasons:
- Some reviews look repeated or suspicious.
- Seller reliability is weak.
- Many reviews contain negative complaints.
- Return policy is unclear.
- Product price is unusually low compared to market price.

## API endpoints
- `GET /health`
- `POST /api/v1/scan`
- `POST /api/v1/feedback`
- `GET /api/v1/model-info`

## MVP quality requirements
- Code should be clean and readable.
- Use type hints in Python.
- Use TypeScript types in extension.
- Do not crash when data is missing.
- Missing data should reduce confidence.
- Do not collect personal user data.
- Use clear risk wording, not legal accusations.
- Add unit tests for scoring engine.

## Build order
1. Backend API skeleton.
2. TrustScore scoring engine.
3. AI/ML inference fallback services.
4. Database schema and logging.
5. Extension data extraction.
6. Popup UI.
7. Feedback submission.
8. Training script.
9. Tests and final README.
