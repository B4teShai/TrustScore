# Datasets and Training Plan

## Purpose
This document explains which open-source datasets can be used and how to train the AI/ML models.

## Recommended datasets

| Dataset | Use | URL |
|---|---|---|
| Fake Reviews Dataset | Fake/real product review classification | https://huggingface.co/datasets/theArijitDas/Fake-Reviews-Dataset |
| Amazon Reviews 2023 | Review text, ratings, item metadata, price, product information | https://amazon-reviews-2023.github.io/main.html |
| Amazon Reviews 2023 on Hugging Face | Easier access to Amazon Reviews 2023 data | https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023 |
| Deceptive Opinion Spam Corpus | Research baseline for deceptive review detection | https://www.kaggle.com/datasets/rtatman/deceptive-opinion-spam-corpus |

## Dataset mapping to model groups

| Model Group | Algorithm | Training Dataset | Notes |
|---|---|---|---|
| Review Sentiment | BERT or DistilBERT | Amazon Reviews 2023 or star ratings as weak labels | Map 1-2 stars to negative, 3 to neutral, 4-5 to positive for prototype |
| Fake Review Detection | Random Forest | Fake Reviews Dataset | Has fake/real labels; useful for supervised training |
| Seller / Price / Policy Risk | Rule-based + ML | Amazon Reviews 2023 + custom collected page data | Price and metadata support risk features; seller/policy may need custom data |

## Fake Reviews Dataset fields
Expected fields:
- `category`
- `rating`
- `text`
- `label`

Label meaning:
- `0`: original/authentic review.
- `1`: computer-generated fake review.

## Training plan: Fake Review Detection Model

### Goal
Train a model that predicts the probability that a review or product review set is suspicious.

### Prototype approach
Train a review-level classifier first, then aggregate review-level predictions to product level.

### Feature extraction
Use a combination of text and metadata features:
- TF-IDF vector features from review text.
- Review length.
- Rating.
- Category.
- Repeated text flag.
- Extreme rating flag.
- Sentiment-rating mismatch if sentiment model is available.

### Model
Use scikit-learn RandomForestClassifier.

### Output
For each review:
- `fake_probability` from 0 to 1.

For each product:
- Average fake probability across reviews.
- `review_authenticity_score = 100 * (1 - avg_fake_probability)`.

### Evaluation metrics
- Accuracy.
- Precision.
- Recall.
- F1 score.
- ROC-AUC, if probability outputs are used.
- Confusion matrix.

## Training plan: Sentiment Model

### Option A: Use existing pre-trained model for MVP
Use a pre-trained Hugging Face sentiment model for fast implementation. Example:
- `distilbert-base-uncased-finetuned-sst-2-english`

This is fast to implement, but trained mostly on general sentiment, not shopping-specific reviews.

### Option B: Fine-tune later
Use Amazon Reviews 2023 with weak labels:
- 1-2 stars: negative.
- 3 stars: neutral.
- 4-5 stars: positive.

### Output
Aggregate review-level sentiment into product-level sentiment score from 0 to 100.

## Training plan: Seller, Price, and Policy Risk

### MVP approach
Use rule-based scoring first.

### Seller reliability example
```text
seller_score = weighted average of:
- seller rating normalized to 0-100
- seller review count confidence adjustment
- years active confidence adjustment
```

### Price safety example
```text
price_ratio = product_price / average_market_price

If average market price is unknown:
    price_safety_score = 50
If price_ratio < 0.50:
    price_safety_score = 35
If 0.50 <= price_ratio < 0.75:
    price_safety_score = 60
If 0.75 <= price_ratio <= 1.30:
    price_safety_score = 90
If price_ratio > 1.30:
    price_safety_score = 70
```

### Policy clarity example
```text
policy_score = 0
If return policy text exists: +30
If refund or return keyword exists: +25
If time period exists, such as 14 days or 30 days: +25
If warranty or exchange keyword exists: +10
If policy text is longer than minimum threshold: +10
Max score = 100
```

## Model artifact outputs
Use the reproducible sampled-data workflow first:

```bash
python ml/training/download_datasets.py \
  --output-dir ml/data \
  --categories All_Beauty \
  --fake-sample-size 20000 \
  --amazon-review-sample-size 20000 \
  --amazon-meta-sample-size 20000
```

Then train all models from local `ml/data` samples:

```bash
python ml/training/train_all.py --skip-download
```

After training, canonical artifacts are saved into grouped folders:

```text
ml/artifacts/
  fake_review/
  sentiment/
  risk/
  training_summary.json
```

The backend also keeps compatibility with older root-level artifact filenames.

Legacy fake-review artifact names:

```text
ml/artifacts/
  fake_review_rf.joblib
  fake_review_vectorizer.joblib
  fake_review_feature_config.json
  model_metadata.json
```

## Model metadata example

```json
{
  "model_name": "fake_review_random_forest",
  "model_version": "0.1.0",
  "trained_at": "2026-05-09T00:00:00Z",
  "dataset": "theArijitDas/Fake-Reviews-Dataset",
  "metrics": {
    "accuracy": 0.0,
    "precision": 0.0,
    "recall": 0.0,
    "f1": 0.0
  }
}
```

## Important dataset limitations
- Fake Reviews Dataset includes generated fake reviews, so it may not represent all real-world fake review behavior.
- Amazon Reviews 2023 is useful for real review data and product metadata, but it is not directly labeled fake/real.
- Seller and return policy fields may need custom extraction from live pages.
- Dataset bias can affect small sellers and new products, so the system should show confidence level.
