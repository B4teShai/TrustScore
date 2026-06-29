# Requirements Summary

## Functional requirements

1. Read the active product-page URL from the browser extension.
2. Preview visible active-tab product data in the popup:
   - title
   - image
   - seller information
   - visible price/rating/review count when present
3. Fetch and extract product data in the backend:
   - title
   - price
   - rating
   - reviews
   - seller information
   - return policy
   - description
4. Send `{ url }` to `POST /api/v1/scan`, then fall back to `POST /api/v1/scan-extracted` only when the retailer blocks backend fetching.
5. Clean and preprocess review data.
6. Extract ML features.
7. Run three model groups:
   - Review Sentiment Model.
   - Fake Review Detection Model.
   - Seller, Price, and Policy Risk Model.
8. Calculate TrustScore from 0 to 100.
9. Classify risk as Low, Medium, or High.
10. Generate top 3 explanation reasons.
11. Show result in popup UI.
12. Collect helpful/not helpful feedback.
13. Store prediction logs and feedback.

## Non-functional requirements

1. Fast response time for product scan.
2. Robust behavior with missing page data.
3. Explainable output.
4. Privacy-preserving data collection.
5. Minimum extension permissions.
6. Maintainable code structure.
7. Unit tests for core scoring logic.
8. No real secrets in repository.

## AI requirements

1. Sentiment score must be 0 to 100.
2. Fake review model must output probability from 0 to 1.
3. Review authenticity score must be 0 to 100.
4. Seller, price, and policy scores must be 0 to 100.
5. TrustScore must be a weighted score from 0 to 100.
6. Explanation must be generated from weakest component scores.
7. Missing data must reduce confidence.

## MVP acceptance criteria

The project is complete when:
- Backend runs locally.
- Extension builds and loads in Chrome.
- Scan endpoint returns realistic TrustScore response.
- Popup displays the result.
- Feedback can be submitted.
- Tests pass for scoring logic.
- Documentation explains setup and architecture.
