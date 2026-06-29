# v3 Leakage Report — risk weak labels

Rows analysed: 180000 (from v2 Amazon metadata sample).

## seller_label

- **Label reconstruction from numeric features alone:** **1.0000** (depth-4 tree).
- Mutual information: {'rating': 0.487, 'log_review_count': 0.6437}
- Leave-one-category-out mean accuracy — **numeric 1.0** vs text-only 0.3953.

Per held-out category:

| Held-out | numeric OOD acc | text-only OOD acc |
|---|---|---|
| All_Beauty | 1.0 | 0.4543 |
| Electronics | 1.0 | 0.3734 |
| Toys_and_Games | 1.0 | 0.3582 |

<details><summary>Recovered decision rule (excerpt)</summary>

```
|--- log_review_count <= 1.70
|   |--- class: weak
|--- log_review_count >  1.70
|   |--- log_review_count <= 3.92
|   |   |--- rating <= 3.45
|   |   |   |--- class: weak
|   |   |--- rating >  3.45
|   |   |   |--- class: mixed
|   |--- log_review_count >  3.92
|   |   |--- rating <= 4.15
|   |   |   |--- rating <= 3.45
|   |   |   |   |--- class: weak
|   |   |   |--- rating >  3.45
|   |   |   |   |--- class: mixed
|   |   |--- rating >  4.15
|   |   |   |--- class: reliable

```
</details>

## price_label

- **Label reconstruction from numeric features alone:** **1.0000** (depth-4 tree).
- Mutual information: {'price': 0.7342, 'price_ratio': 0.979}
- Leave-one-category-out mean accuracy — **numeric 1.0** vs text-only 0.5431.

Per held-out category:

| Held-out | numeric OOD acc | text-only OOD acc |
|---|---|---|
| All_Beauty | 0.9999 | 0.5989 |
| Electronics | 1.0 | 0.4842 |
| Toys_and_Games | 1.0 | 0.5463 |

<details><summary>Recovered decision rule (excerpt)</summary>

```
|--- price_ratio <= 1.80
|   |--- price_ratio <= 0.50
|   |   |--- class: suspicious_low
|   |--- price_ratio >  0.50
|   |   |--- price_ratio <= 1.80
|   |   |   |--- class: normal
|   |   |--- price_ratio >  1.80
|   |   |   |--- price <= 37.76
|   |   |   |   |--- class: high_price
|   |   |   |--- price >  37.76
|   |   |   |   |--- class: normal
|--- price_ratio >  1.80
|   |--- class: high_price

```
</details>

## Verdict

The numeric features **reconstruct the label almost perfectly** and the recovered tree recovers the exact thresholds used to *construct* the weak label (rating/review-count for seller; price ratio for price). Because the rule is global, the numeric model stays high even on a held-out category — that is **rule reconstruction, not trust generalisation**.

**We reject the v2 in-sample ~0.99 seller/price accuracy as a trust metric.** v3 reports the unsupervised price-anomaly signal and out-of-domain numbers instead (see FINDINGS.md).
