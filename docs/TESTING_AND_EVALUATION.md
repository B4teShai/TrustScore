# Testing and Evaluation Plan

## Backend tests

Use pytest.

### Unit tests
Test:
- Preprocessing removes empty reviews.
- Duplicate review detection works.
- Sentiment aggregation returns 0 to 100.
- Fake review fallback returns 0 to 100.
- Seller score handles missing values.
- Price score handles missing average market price.
- Policy score detects return/refund words.
- TrustScore formula is correct.
- Risk classification thresholds are correct.
- Explanation generator returns top 3 reasons.

### API tests
Test:
- `GET /health` returns status ok.
- `POST /api/v1/scan` accepts valid payload.
- `POST /api/v1/scan` handles missing optional fields.
- `POST /api/v1/feedback` stores feedback.
- Invalid payload returns validation error.

## Extension tests

Use TypeScript type checking and optional Vitest.

Test:
- Extractors return `ProductScanPayload` shape.
- API client handles success response.
- API client handles structured backend errors and invalid response payloads.
- Manifest declares required local backend host permissions.
- Popup storage helpers persist the last result and create browser IDs only for feedback.
- Risk label renders correct text.
- Score badge renders low/medium/high state.

## Model evaluation

### Fake Review Detection Model
Metrics:
- Accuracy.
- Precision.
- Recall.
- F1 score.
- ROC-AUC.
- Confusion matrix.

Important: prioritize recall if the goal is warning users about suspicious reviews, but keep false positives low to avoid unfairly warning against good products.

### Sentiment Model
Metrics:
- Accuracy on labeled validation data.
- Macro F1 if using positive, neutral, negative labels.
- Manual review of confusing examples.

### TrustScore output
Because TrustScore is a combined risk score, evaluate it with:
- Manual test cases.
- User feedback helpfulness rate.
- Component score sanity checks.

## Example manual test cases

### Case 1: Low risk product
Input:
- Many reviews.
- Mostly positive sentiment.
- Few duplicates.
- Reliable seller.
- Normal price.
- Clear return policy.

Expected:
- TrustScore >= 80.
- Risk Level: Low Risk.

### Case 2: Medium risk product
Input:
- Some repeated reviews.
- Mixed sentiment.
- Seller information exists but policy unclear.

Expected:
- TrustScore 50 to 79.
- Risk Level: Medium Risk.

### Case 3: High risk product
Input:
- Many duplicate short 5-star reviews.
- Negative text/rating mismatch.
- Unknown seller.
- Suspicious low price.
- No return policy.

Expected:
- TrustScore below 50.
- Risk Level: High Risk.

## Example pytest cases

```python
def test_risk_classification():
    assert classify_risk(85) == "Low Risk"
    assert classify_risk(50) == "Medium Risk"
    assert classify_risk(49) == "High Risk"


def test_trustscore_formula():
    scores = {
        "review_authenticity": 100,
        "seller_reliability": 50,
        "sentiment": 50,
        "return_policy_clarity": 50,
        "price_safety": 50,
        "user_feedback_history": 50,
    }
    result = calculate_trustscore(scores)
    assert result == 65
```

## Demo acceptance checklist

- Backend starts.
- Extension builds.
- Extension `npm test -- --run` passes.
- Extension popup opens.
- Sample product page can be analyzed.
- TrustScore is visible.
- Top reasons are visible.
- User feedback button works.
- System does not collect personal data.
