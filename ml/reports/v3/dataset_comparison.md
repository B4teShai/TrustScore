# v3 Dataset Comparison & Recommendation

Quality score = 0.4·class-balance + 0.3·size + 0.3·license-openness.

| Rank | Dataset | Task | Rows | Balance | License | Openness | Quality |
|---|---|---|---|---|---|---|---|
| 1 | amazon_polarity | sentiment | 100000 | 0.986 | Apache-2.0 (HF card) / Amazon review data terms | 0.9 | **0.964** |
| 2 | yelp_polarity | sentiment | 100000 | 0.873 | Yelp Dataset terms (research/personal) | 0.6 | **0.829** |
| 3 | sst2 | sentiment | 67349 | 0.793 | Permissive (GLUE/SST-2) | 1.0 | **0.819** |
| 4 | fake_review_cross | fake_review | 40432 | 1.0 | Research use (Salminen et al. fake reviews) | 0.6 | **0.701** |
| 5 | fake_reviews_v2 | fake_review | 40526 | 0.997 | Research use (theArijitDas) | 0.6 | **0.7** |
| 6 | imdb | sentiment | 25000 | 1.0 | Research use (Maas et al. 2011) | 0.6 | **0.655** |
| 7 | amazon_reviews_v2 | sentiment | 450000 | 0.0 | Amazon review data terms | 0.4 | **0.42** |
| 8 | amazon_meta_v2 | risk | 180000 | 0.0 | Amazon review data terms | 0.4 | **0.42** |

## Recommendation by task

- **sentiment:** amazon_polarity (quality 0.964, 100000 rows, Apache-2.0 (HF card) / Amazon review data terms)
- **fake_review:** fake_review_cross (quality 0.701, 40432 rows, Research use (Salminen et al. fake reviews))
- **risk:** amazon_meta_v2 (quality 0.42, 180000 rows, Amazon review data terms)

## Gated / manual (documented, not downloaded)

- **yelpchi** — YelpChi (Rayana & Akoglu) requires author request; not redistributable on HF.
- **yelp_full** — Full Yelp Open Dataset requires accepting Yelp's dataset agreement; manual download.
- **ieee_cis_fraud** — Kaggle competition data; license requires Kaggle login + competition rules acceptance.

## Unavailable (HF ids did not resolve)

- ott_deceptive (fake_review) — tried ['lukesjordan/deceptive-opinion-spam', 'kkfromus/deceptive-opinion-spam', 'deceptive-opinion']
