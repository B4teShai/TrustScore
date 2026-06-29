# v3 Risk Scorer — leakage-free (Stage 0 fix)

Replaces the rejected v2 weak-label seller/price classifier.

## Approach

- **price_safety**: unsupervised IsolationForest on category-normalised price (no label).
- **seller_reliability**: transparent deterministic rule (rating / review-count / tenure).
- **return_policy_clarity**: transparent deterministic rule (return/period/warranty signals).

## Why this is leakage-free

No model is trained on the v2 weak labels (which were functions of the features). Price uses unsupervised anomaly detection; seller/policy use rules.

## Price-anomaly validation (no labels → injected extremes)

- Injected-anomaly recall: **1.0** (synthetic extreme prices flagged).
- Normal false-flag rate: 0.124.

Seller and policy scores are deterministic rules, so they need no statistical validation; they are interpretable by inspection. A leakage-gate test (`test_leakage_gate*`) guards against any future supervised risk model that trains on its own label's inputs.

