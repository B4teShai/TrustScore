# v3 Feature Analysis — fake review

## SHAP — linguistic features (interpretable)

Mean |SHAP| from an XGBoost on linguistic features only (figure: `figures/shap_fake_linguistic.png`).

| Feature | mean \|SHAP\| |
|---|---|
| type_token_ratio | 1.9956 |
| char_len | 1.4934 |
| avg_word_len | 0.6168 |
| word_len | 0.491 |
| punct_ratio | 0.2113 |
| exclaim_count | 0.1622 |
| caps_ratio | 0.0 |

## Ablation — feature-set value (LogReg)

| Feature set | Accuracy | ROC-AUC |
|---|---|---|
| linguistic_only | 0.7588 | 0.8485 |
| tfidf_only | 0.9258 | 0.9762 |
| embeddings_only | 0.7658 | 0.8515 |
| embeddings_plus_linguistic | 0.8242 | 0.9085 |

## Documented feature gaps (data does not support)

- **reviewer_history**: No per-reviewer id in the review datasets used.
- **rating_velocity**: No time-series of ratings per seller/product in the metadata.
- **seller_age**: No seller registration/first-seen date in the metadata.
- **review_growth**: No longitudinal review-count snapshots available.
