# TrustScore — Final Production Model Set

The chosen model for each of the five signals, after v1→v2→v3. Paths span v2 and v3 (no duplication). All are leakage-safe.

| Signal | Final model | Ver | Metric | Artifact |
|---|---|---|---|---|
| review_authenticity | calibrated LinearSVC + word/char TF-IDF | v3 | ROC-AUC 0.989 / acc 0.949 (in-domain), 0.91 cross-category | `artifacts/v3/fake_review/model.joblib` |
| sentiment | TF-IDF + LogReg (neg/neu/pos) | v2 | acc 0.867 (weak star labels) | `artifacts/v2/sentiment/sentiment_tfidf_logreg.joblib` |
| seller_reliability | transparent rule (rating/reviews/tenure) | v3 | deterministic | `(rule, no artifact)` |
| price_safety | unsupervised IsolationForest anomaly + ratio rule | v3 | injected-anomaly recall 1.0 | `artifacts/v3/risk/price_anomaly_iforest.joblib` |
| return_policy_clarity | transparent rule (return/period/warranty) | v3 | deterministic | `(rule, no artifact)` |

## End-to-end finalize check (demo product: $8 vs $60 market, weak seller, no returns)

All chosen artifacts loaded and scored together:

| Component | Score |
|---|---|
| review_authenticity | 97 |
| sentiment | 90 |
| seller_reliability | 40 |
| price_safety | 25 |
| return_policy_clarity | 30 |
| user_feedback_history | 50 |

**TrustScore = 65 → Medium Risk** (weighted: review_authenticity 0.3, seller_reliability 0.2, sentiment 0.2, return_policy_clarity 0.15, price_safety 0.1, user_feedback_history 0.05).

## Verdict

No retraining required — fake-review (v3 winner) and risk (v3 leakage-free) are freshly trained by the v3 pipeline; sentiment uses the validated v2 model; seller/policy are transparent rules. The finalize gate (`finalize.py`) fails if any chosen artifact is missing.
