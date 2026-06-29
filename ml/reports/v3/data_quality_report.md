# v3 Data Quality Report

| Dataset | Rows | Empty | Exact dup | Near dup | Imbalance | Mean chars |
|---|---|---|---|---|---|---|
| fake_reviews_v2 | 40000 | 0.0 | 0.0008 | 0.001 | 0.995 | 350.6 |
| fake_review_cross | 40000 | 0.0 | 0.001 | 0.001 | 0.999 | 347.9 |
| amazon_polarity | 40000 | 0.0 | 0.0 | 0.0 | 0.977 | 432.1 |
| yelp_polarity | 40000 | 0.0 | 0.0 | 0.0 | 0.876 | 703.9 |
| imdb | 25000 | 0.0 | 0.0039 | 0.0039 | 1.0 | 1302.3 |
| sst2 | 40000 | 0.0 | 0.003 | 0.003 | 0.792 | 52.5 |
| amazon_reviews_v2 | 40000 | 0.0003 | 0.0391 | 0.0439 | - | 266.5 |

## Distribution shift (text length, KS vs Amazon)

- amazon_vs_yelp_polarity_textlen_KS: **0.2121** (0 = identical, 1 = maximal shift)
- amazon_vs_imdb_textlen_KS: **0.6198** (0 = identical, 1 = maximal shift)
- amazon_vs_sst2_textlen_KS: **0.8867** (0 = identical, 1 = maximal shift)

## Label-noise probe

- fake_reviews_v2: confident-misclassification rate **0.0012** — High-confidence disagreements are candidate mislabels or genuinely ambiguous items.
