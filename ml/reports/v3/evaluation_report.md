# v3 Real-World Evaluation

Headline metric: **generalisation under distribution shift**, not in-sample accuracy.

## Sentiment — cross-domain (train = Amazon polarity)

| Target | n | Acc | F1 | ROC-AUC |
|---|---|---|---|---|
| amazon (in-domain) | 8000 | 0.8985 | 0.8992 | 0.9647 |
| yelp_polarity (OOD) | 40000 | 0.8889 | 0.8813 | 0.957 |
| imdb (OOD) | 25000 | 0.8584 | 0.8509 | 0.9405 |
| sst2 (OOD) | 40000 | 0.7239 | 0.7708 | 0.7977 |

The accuracy drop from Amazon to Yelp/IMDB/SST-2 quantifies real domain shift.

## Fake review — leave-one-category-out (genuine OOD)

- In-domain random split: acc **0.9346**, AUC 0.9831.
- Mean leave-one-category-out accuracy: **0.9102**.

| Held-out category | n | Acc | ROC-AUC |
|---|---|---|---|
| Kindle_Store | 4730 | 0.9156 | 0.9758 |
| Books | 4370 | 0.9048 | 0.975 |
| Pet_Supplies | 4254 | 0.9031 | 0.9742 |
| Home_and_Kitchen | 4056 | 0.9406 | 0.9852 |
| Electronics | 3988 | 0.9233 | 0.9814 |
| Sports_and_Outdoors | 3946 | 0.9306 | 0.9836 |
| Tools_and_Home_Improvement | 3858 | 0.93 | 0.9827 |
| Clothing_Shoes_and_Jewelry | 3848 | 0.9015 | 0.9646 |
| Toys_and_Games | 3794 | 0.8919 | 0.9713 |
| Movies_and_TV | 3588 | 0.8601 | 0.9658 |

## Calibration (fake-review probe)

- Raw: Brier 0.058, ECE 0.0782.
- Isotonic: Brier 0.0538, ECE 0.0073.
- Reliability diagram: `figures/calibration_reliability.png`.

## Robustness (text perturbations)

Baseline accuracy 0.9323.

| Perturbation | Acc | Δ |
|---|---|---|
| lowercase | 0.9323 | 0.0 |
| truncate_50_chars | 0.6949 | -0.2374 |
| drop_vowels_10pct | 0.9096 | -0.0226 |

## Risk — out-of-domain (from leakage report)

Numeric risk models keep ~1.0 accuracy leave-one-category-out because the label is a global rule of the numeric features (leakage). Text-only collapses. See `leakage_report.md`. v3 therefore treats risk via the unsupervised price-anomaly signal, not the weak-label classifier.

