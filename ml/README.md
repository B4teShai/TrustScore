# TrustScore ML Training Guide

This folder contains the training code for the TrustScore submodels. Run all
commands from the repository root:

```bash
cd /Users/altanshagai/Desktop/side/TrustScore
```

The backend already works without trained artifacts. Training is only needed
when you want to replace heuristic fallback scoring with saved model files.

## 1. Install Training Dependencies

Use the same Python environment you use for the backend, or create a fresh one:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r apps/api/requirements.txt
pip install -r ml/requirements.txt
```

Check that the training scripts are callable:

```bash
python ml/training/train_fake_review_model.py --help
python ml/training/train_sentiment_model.py --help
python ml/training/train_risk_models.py --help
```

## 2. Run Local Smoke Tests First

This validates the training code without downloading real datasets:

```bash
python -m pytest ml/tests
```

Expected result:

```text
4 passed
```

## 3. How Dataset Download Works

Use the downloader when you want reproducible local samples under `ml/data`:

```bash
python ml/training/download_datasets.py \
  --output-dir ml/data \
  --categories All_Beauty \
  --fake-sample-size 20000 \
  --amazon-review-sample-size 20000 \
  --amazon-meta-sample-size 20000
```

The Hugging Face datasets can also download automatically when you run a script
without a local CSV flag.

For example, this command starts downloading
`theArijitDas/Fake-Reviews-Dataset`:

```bash
python ml/training/train_fake_review_model.py \
  --sample-size 20000 \
  --output-dir ml/artifacts
```

The dataset cache is managed by the `datasets` library. If you want a specific
cache location:

```bash
HF_HOME=ml/data/huggingface_cache python ml/training/train_fake_review_model.py \
  --sample-size 20000 \
  --output-dir ml/artifacts
```

## 4. Train Fake-Review Authenticity Model

Purpose:
- Predict fake-review probability.
- Produce `review_authenticity` for the TrustScore backend.

Dataset:
- `theArijitDas/Fake-Reviews-Dataset`
- Fields: `category`, `rating`, `text`, `label`
- Label `0`: original/authentic
- Label `1`: generated fake review

Fast training:

```bash
python ml/training/train_fake_review_model.py \
  --sample-size 20000 \
  --n-estimators 300 \
  --output-dir ml/artifacts
```

Full available dataset:

```bash
python ml/training/train_fake_review_model.py \
  --n-estimators 500 \
  --output-dir ml/artifacts
```

Artifacts written:

```text
ml/artifacts/fake_review_rf.joblib
ml/artifacts/fake_review_vectorizer.joblib
ml/artifacts/fake_review_feature_config.json
ml/artifacts/model_metadata.json
```

These filenames match the FastAPI backend loader.

## 5. Train Review Sentiment Model

Purpose:
- Predict review sentiment from review text.
- Produce the `sentiment` component for TrustScore.

Dataset:
- `McAuley-Lab/Amazon-Reviews-2023`
- The loader reads explicit `raw/review_categories/*.jsonl` files with the
  standard Hugging Face `json` builder. It does not use the repository's legacy
  dataset script or `trust_remote_code`.
- Uses Amazon review stars as weak labels:
  - `1-2`: negative
  - `3`: neutral
  - `4-5`: positive

Start with a small category and sample:

```bash
python ml/training/train_sentiment_model.py \
  --categories All_Beauty \
  --sample-size 20000 \
  --output-dir ml/artifacts
```

Then train on more shopping categories:

```bash
python ml/training/train_sentiment_model.py \
  --categories All_Beauty Electronics \
  --sample-size 50000 \
  --output-dir ml/artifacts
```

Artifact written:

```text
ml/artifacts/sentiment_tfidf_logreg.joblib
ml/artifacts/sentiment_model_metadata.json
```

Note: the current backend still uses the existing runtime sentiment fallback.
This artifact is prepared for the next integration step.

## 6. Train Seller, Price, and Policy Risk Models

Purpose:
- Prepare ML models for `seller_reliability`, `price_safety`, and
  `return_policy_clarity`.

Datasets:
- Amazon Reviews 2023 metadata for seller and price signals:
  - Loaded from explicit `raw_meta_*/*.parquet` files with the standard
    Hugging Face `parquet` builder, not the repository's legacy dataset script.
  - `store`
  - `average_rating`
  - `rating_number`
  - `price`
  - `main_category`
- Target product sample CSV for return-policy text:
  - `shipping_returns_policy`

Amazon metadata downloads automatically through the script:

```bash
python ml/training/train_risk_models.py \
  --categories All_Beauty Electronics \
  --sample-size 50000 \
  --output-dir ml/artifacts
```

For policy clarity, first place a Target product CSV at:

```text
ml/data/target-products.csv
```

It must include this column:

```text
shipping_returns_policy
```

Then run:

```bash
python ml/training/train_risk_models.py \
  --categories All_Beauty Electronics \
  --policy-csv ml/data/target-products.csv \
  --sample-size 50000 \
  --output-dir ml/artifacts
```

Artifacts written:

```text
ml/artifacts/seller_reliability_tfidf_rf.joblib
ml/artifacts/price_safety_tfidf_rf.joblib
ml/artifacts/policy_clarity_tfidf_rf.joblib
ml/artifacts/risk_model_metadata.json
```

Note: these risk artifacts are not wired into the backend yet. The backend still
uses deterministic rule-based seller, price, and policy scoring.

## 7. Optional TrustScore Calibration

Public datasets do not provide a real final `trust_score` label. Do not invent
one.

Only run calibration if you create a human-labeled CSV with these columns:

```text
review_authenticity
seller_reliability
sentiment
return_policy_clarity
price_safety
user_feedback_history
trust_score
```

Command:

```bash
python ml/training/calibrate_trustscore.py \
  --local-csv ml/data/human_trustscore_labels.csv \
  --output-dir ml/artifacts
```

If no CSV is passed, the script safely skips calibration:

```bash
python ml/training/calibrate_trustscore.py --no-save
```

## 8. Use Local CSV Instead of Downloading

Fake-review local CSV:

```bash
python ml/training/train_fake_review_model.py \
  --local-csv ml/data/fake_reviews.csv \
  --output-dir ml/artifacts
```

Required columns:

```text
text,label
```

Sentiment local CSV:

```bash
python ml/training/train_sentiment_model.py \
  --local-csv ml/data/amazon_reviews_sample.csv \
  --output-dir ml/artifacts
```

Required columns:

```text
rating,text
```

Risk local CSV:

```bash
python ml/training/train_risk_models.py \
  --amazon-meta-csv ml/data/amazon_meta_sample.csv \
  --policy-csv ml/data/target-products.csv \
  --output-dir ml/artifacts
```

Required Amazon metadata columns:

```text
title,average_rating,rating_number,price
```

Optional but useful:

```text
store,main_category
```

## 9. Recommended Training Order

Run in this order:

```bash
python ml/training/download_datasets.py --output-dir ml/data
```

```bash
python ml/training/train_all.py --skip-download
```

Or run the individual commands below when you need more control:

```bash
python -m pytest ml/tests
```

```bash
python ml/training/train_fake_review_model.py \
  --sample-size 20000 \
  --output-dir ml/artifacts
```

```bash
python ml/training/train_sentiment_model.py \
  --categories All_Beauty \
  --sample-size 20000 \
  --output-dir ml/artifacts
```

```bash
python ml/training/train_risk_models.py \
  --categories All_Beauty \
  --sample-size 20000 \
  --output-dir ml/artifacts
```

After fake-review training, verify the backend sees the artifact:

```bash
cd apps/api
python -m pytest
```

Then start the backend:

```bash
uvicorn app.main:app --reload --port 8000
```

Check model status:

```bash
curl http://localhost:8000/api/v1/model-info
```

Look for:

```json
{
  "fake_review_artifact_status": "loaded"
}
```

## 10. Important Notes

- Start with `--sample-size` so downloads and training are faster.
- Remove `--sample-size` only after the small run succeeds.
- Do not commit large downloaded datasets or generated artifacts unless the
  project explicitly needs them.
- The final TrustScore is not directly trained from public datasets yet. It is
  calculated from submodel outputs using the backend weighted formula.

## 11. v2 Training (second iteration)

v2 is a separate, versioned pipeline under `ml/training/v2/`. It never touches the
v1 scripts or `ml/artifacts/` — v2 artifacts go to `ml/artifacts/v2/` with
`model_version` `0.2.0`, and reports to `ml/reports/v2/`.

What changed in v2:

- **Risk models (headline):** seller/price/policy labels are functions of numeric
  metadata (rating, review count, price ratio) and policy-text flags. v1 fed text
  only and scored ~0.58/0.60. v2 adds those numeric features via a
  `ColumnTransformer` (numeric + TF-IDF text) and reports a text-only-vs-+numeric
  ablation. The previously skipped **return-policy** model is now trained.
- **Fake review:** word + `char_wb` TF-IDF, model selection across
  LogReg / calibrated LinearSVC / ComplementNB / RandomForest (best by ROC-AUC).
- **Sentiment:** multi-category data, word + char TF-IDF, LogReg vs calibrated LinearSVC.
- Backend-compatible artifact interfaces are preserved (fake = vectorizer + binary
  model pair; sentiment = `negative/neutral/positive` pipeline; risk = DataFrame
  pipeline + sibling `*_feature_spec.json`).

Reproduce (datasets must already be downloaded into `ml/data/`):

```bash
# 1. download full datasets (valid metadata categories only)
python ml/training/download_datasets.py --output-dir ml/data \
  --categories All_Beauty Electronics Toys_and_Games \
  --fake-sample-size 100000 --amazon-review-sample-size 150000 \
  --amazon-meta-sample-size 60000 --target-policy-sample-size 10000

# 2. run the full v2 pipeline: dataset report -> baselines -> train -> compare
PYTHONPATH=. python ml/training/v2/train_all_v2.py

# outputs:
#   ml/artifacts/v2/...                  (versioned models + metadata)
#   ml/reports/v2/dataset_report.json    (+ figures/)
#   ml/reports/v2/baselines.json
#   ml/reports/v2/comparison_v1_v2.json  (+ figures/comparison_accuracy.png)
```

Serve v2 from the backend (v1 stays the default):

```bash
cd apps/api
TRUSTSCORE_MODEL_VERSION=v2 uvicorn app.main:app --port 8000
curl http://localhost:8000/api/v1/model-info   # -> model_version 0.2.0
```

Smoke-test the v2 trainers without large data:

```bash
PYTHONPATH=. python -m pytest ml/tests/test_v2_training.py
```

## 12. v3 Research branch (model_version 0.3.0)

v3 is a **research-grade, honesty-first** branch, fully isolated in `ml/training/v3/`,
`ml/artifacts/v3/`, `ml/reports/v3/`, `ml/data/v3/`. It never touches v1/v2.

Its purpose is **not** higher accuracy. v2 flagged that the seller/price risk "wins" are
inflated by weak-label leakage; v3 proves it, measures real generalisation, and rejects
leakage shortcuts. Highlights (`ml/reports/v3/FINDINGS.md`):

- **Leakage proved & rejected:** numeric-only models reconstruct the seller/price labels at
  1.00 and stay 1.00 leave-one-category-out (rule reconstruction, not trust). Risk is reframed
  as an **unsupervised price-anomaly** signal.
- **Fake review (genuine):** calibrated LinearSVC on word+char TF-IDF — ROC-AUC 0.989, and
  0.910 mean cross-category OOD. **TF-IDF beat frozen transformer embeddings** (reported honestly).
- **Sentiment domain shift:** Amazon→Yelp 0.889, →IMDB 0.858, →SST-2 0.724.
- **Calibration:** isotonic cuts ECE 0.078→0.007. **Robustness:** fragile to 50-char truncation (−24pp).

Pipeline: dataset comparison + licensing, data-quality + leakage + bias audits, SHAP +
ablation, full algorithm leaderboard (classical + XGBoost/LightGBM/CatBoost + embeddings +
ensembles), OOD/calibration/robustness evaluation, extraction audit, experiment registry +
model cards + reproducibility report.

```bash
# acquire curated HF datasets (capped), then run the full pipeline
PYTHONPATH=. python ml/training/v3/data_sources/acquire.py
PYTHONPATH=. python ml/training/v3/run_v3.py          # each phase also runs standalone
PYTHONPATH=. python -m pytest ml/tests/test_v3.py
```

- **Risk fix (leakage-free):** `risk_v3.py` replaces the rejected v2 risk classifier with an
  unsupervised price-anomaly detector + transparent seller/policy rules; a leakage gate
  (`quality/leakage_gate.py`) blocks any future model that trains on its label's own inputs.
- **Final production set:** `finalize.py` confirms the chosen model per signal loads and scores
  end-to-end (`PRODUCTION_MODELS.md`). No retraining needed — fake/risk are v3-fresh, sentiment is v2.

Reports: `PRODUCTION_MODELS.md`, `RESULTS.md`, `FINDINGS.md`, `leakage_report.md`, `risk_v3.md`,
`leaderboard.md`, `evaluation_report.md`, `dataset_comparison.md`, `data_quality_report.md`,
`bias_report.md`, `feature_importance_v3.md`, `extraction_audit.md`, `reproducibility_report.md`,
plus `model_cards/` and `figures/`.

## Best artifact set (`ml/artifacts/best/`)

`make_best_artifacts.py` gathers the single best leakage-safe model per signal into one
curated directory, alongside a `best_manifest.json`. Version-by-version artifacts stay under
`ml/artifacts/{v1,v2,v3}`.

```bash
PYTHONPATH=. python -m ml.training.make_best_artifacts
```

Serve it: `TRUSTSCORE_MODEL_VERSION=best` (model_version `1.0.0`). The Docker stack runs this
assembler in the `trainer` service on startup, so the API always boots with the best set.
Run the full training pipeline on demand inside Docker:

```bash
docker compose run --rm trainer python -m ml.training.train_all
```
