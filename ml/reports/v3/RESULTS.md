# TrustScore v3 — Results (digest)

One-glance summary; full detail in FINDINGS.md and the per-phase reports.

## 1. Risk weak-label leakage (rejected)

| Label | numeric reconstruction | numeric OOD | text-only OOD |
|---|---|---|---|
| price_label | 1.0 | 1.0 | 0.5431 |
| seller_label | 1.0 | 1.0 | 0.3953 |

Numeric models reconstruct the label (~1.0) and keep ~1.0 out-of-category while text-only collapses → **leakage, rejected as a trust metric**.

### Fix (Stage 0) — leakage-free risk scorer

- **Price safety:** unsupervised IsolationForest anomaly (no label) — injected-anomaly recall **1.0**, false-flag 0.124.
- **Seller / policy:** transparent deterministic rules (no trained classifier).
- A **leakage-gate test** blocks any future model that trains on its label's own inputs.

## 2. Fake-review leaderboard

**Winner:** `calibrated_linear_svc` on `tfidf` — ROC-AUC **0.9889**, acc 0.9492, 704.8 KB.

| Model | Features | Acc | ROC-AUC | Size KB |
|---|---|---|---|---|
| calibrated_linear_svc | tfidf | 0.9492 | 0.9889 | 704.8 |
| lightgbm | tfidf | 0.9419 | 0.9872 | 2071.1 |
| logreg | tfidf | 0.9425 | 0.9854 | 235.1 |
| xgboost | tfidf | 0.9218 | 0.98 | 812.7 |
| random_forest | tfidf | 0.8979 | 0.9606 | 146632.2 |

## 3. Feature ablation (fake review)

| Feature set | Acc | ROC-AUC |
|---|---|---|
| linguistic_only | 0.7588 | 0.8485 |
| tfidf_only | 0.9258 | 0.9762 |
| embeddings_only | 0.7658 | 0.8515 |
| embeddings_plus_linguistic | 0.8242 | 0.9085 |

TF-IDF beats frozen embeddings — lexical signal dominates (reported honestly).

## 4. Out-of-domain generalisation

- Sentiment (train Amazon, acc 0.8985): imdb 0.8584, sst2 0.7239, yelp_polarity 0.8889.
- Fake review: in-domain 0.9346, mean leave-one-category-out **0.9102**.

## 5. Calibration & robustness

- Calibration ECE: raw 0.0782 → isotonic **0.0073**.
- Robustness (Δacc): drop_vowels_10pct -0.0226, lowercase 0.0, truncate_50_chars -0.2374.

