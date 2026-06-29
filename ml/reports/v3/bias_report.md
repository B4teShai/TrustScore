# v3 Bias Report

## Class balance

- **fake_reviews_v2**: imbalance ratio 0.995 (1.0 = perfectly balanced).
- **fake_review_cross**: imbalance ratio 0.999 (1.0 = perfectly balanced).
- **amazon_polarity**: imbalance ratio 0.977 (1.0 = perfectly balanced).
- **yelp_polarity**: imbalance ratio 0.876 (1.0 = perfectly balanced).
- **imdb**: imbalance ratio 1.0 (1.0 = perfectly balanced).
- **sst2**: imbalance ratio 0.792 (1.0 = perfectly balanced).

## Domain & coverage bias

- All text datasets are **English only** — non-English reviews are out of scope.
- Sentiment weak labels in v2 (`amazon_reviews_v2`) are ~80% positive (star-rating skew); the v3 sentiment study uses balanced polarity corpora instead.
- v2 risk metadata is dominated by a few categories (All Beauty / Toys), biasing category-median price baselines; v3 reports category-normalised features.
- Fake-review corpora are product/hotel reviews; generalisation to other review types is untested.

## Distribution shift

Non-trivial text-length shift between Amazon and Yelp/IMDB/SST-2 (see data_quality_report.md) means a model trained on one domain will degrade on another — quantified in evaluation_report.md.
