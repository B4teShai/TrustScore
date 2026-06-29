# TrustScore v3 — Research Findings & Recommendations

**Branch:** v3 (model_version 0.3.0), isolated in `ml/training/v3/`, `ml/artifacts/v3/`,
`ml/reports/v3/`. v1/v2 untouched. **Goal: trustworthy, defensible results — not inflated accuracy.**

---

## TL;DR

1. **The v2 risk "99% accuracy" is a leakage artifact — rejected.** The seller/price labels
   are deterministic functions of the numeric features fed to the model; a depth-4 tree on
   numeric features alone reconstructs the label at **1.00**, and the model stays at **1.00**
   leave-one-category-out (rule reconstruction, not trust generalisation). v3 replaces the
   weak-label risk classifier with an **unsupervised price-anomaly** signal.
2. **Fake-review detection is genuine and generalises.** Best model (calibrated LinearSVC on
   word+char TF-IDF) scores **ROC-AUC 0.989 / acc 0.949** in-domain and **0.910** mean
   leave-one-category-out — only ~2.4pp drop. This is real signal, not leakage.
3. **TF-IDF beats frozen transformer embeddings here** (0.95 vs 0.82–0.84). The DL track did
   NOT win; we report that honestly rather than forcing a transformer.
4. **Sentiment shows real domain shift:** train Amazon → 0.899 in-domain, 0.889 Yelp,
   0.858 IMDB, **0.724 SST-2** (short sentences; KS text-length shift 0.89).
5. **Calibration matters:** isotonic calibration cuts ECE from 0.078 → **0.007**.
6. **Robustness gap:** the fake model is robust to casing/vowel-drop but loses **23.7pp** under
   50-char truncation — a real fragility to short inputs.

---

## Phase 1 — datasets
Acquired (HF, capped, balanced): amazon_polarity (100k), yelp_polarity (100k), imdb (25k),
sst2 (67k), plus a second fake-review corpus `fake_review_cross` (40k, with category labels →
enables genuine cross-category OOD). Documented gated/manual sets (YelpChi, full Yelp,
IEEE-CIS fraud) and the unavailable Ott ids. Ranking + licensing in `dataset_comparison.md`.
Note: `fake_review_cross` shares lineage with the v1/v2 fake set, so we use **cross-category**
holdout (not cross-source) as the honest OOD test.

## Phase 2 — data quality & leakage
- Clean data: exact/near-duplicate rates <0.5% for the curated sets; fake-review label-noise
  probe (confident-misclassification) **0.0012**. (`data_quality_report.md`, `bias_report.md`)
- **Leakage (centerpiece, `leakage_report.md`):** numeric-only reconstruction = 1.00 for both
  seller and price; mutual information price_ratio→price_label = 0.98; leave-one-category-out
  numeric accuracy = 1.00 vs text-only 0.40 (seller) / 0.54 (price). **Verdict: the v2 risk
  metric is rejected as a measure of trust.**

## Phase 3 — features (`feature_importance_v3.md`)
- SHAP on interpretable linguistic features: `type_token_ratio` and `char_len` dominate.
- Ablation (fake review): tfidf **0.926** > embeddings+linguistic 0.824 > embeddings 0.766 ≈
  linguistic 0.759. Lexical n-grams carry the signal; embeddings add nothing here.
- Documented feature gaps (data cannot support): reviewer history, rating velocity, seller age,
  review growth — not fabricated.

## Phase 4 — algorithm leaderboard (`leaderboard.md`)
Full field on fake review (classical + XGBoost/LightGBM/CatBoost + frozen-embedding head +
voting/stacking). Winner by ROC-AUC **and** deployment cost: **calibrated LinearSVC on TF-IDF**
— AUC 0.989, 705 KB, ~1s train. RandomForest matched older numbers but produced a 146 MB
artifact (rejected on cost). Embedding-based models trailed.

## Phase 6 — real-world evaluation (`evaluation_report.md`)
- Cross-domain sentiment and cross-category fake review as above.
- Calibration: raw Brier 0.058 / ECE 0.078 → isotonic 0.054 / **0.007**.
- Robustness: lowercase Δ0.000, vowel-drop Δ−0.023, **truncate-50 Δ−0.237**.

## Phase 5 — extraction (proposal, `extraction_audit.md`)
Audited `product_page.py` + `fetcher.py`; proposed validation/normalisation, currency
handling, explicit missing-field signals, error-recovery taxonomy, and a coverage harness
(`extraction_coverage.py`, demonstrated). No backend rewrite (per scope).

## Phase 7 — production readiness
Experiment registry (`experiments.jsonl`), model cards (`model_cards/`), dataset version
hashes (`dataset_versions.json`), reproducibility report, typed/documented modules, v3 tests.

---

## Rejected approaches (explicitly flagged)
- **Weak-label risk classifier (v2 style):** rejected — leakage shortcut (proved at 1.00
  reconstruction / 1.00 LOCO). Use unsupervised price anomaly + transparent rules instead.
- **Frozen-embedding fake-review model:** not adopted as the winner — underperformed TF-IDF
  on this task; kept only as a documented comparison.
- **Claiming cross-source fake-review generalisation:** avoided — the second corpus shares
  lineage; we report cross-**category** OOD only.

## Fix shipped (Stage 0) — `risk_v3.py`, `risk_v3.md`
The rejected risk classifier is **replaced**, not just flagged:
- **Price safety:** unsupervised IsolationForest anomaly on category-normalised price (no label
  used → nothing to leak); injected-anomaly recall **1.0**, normal false-flag 0.12.
- **Seller reliability & policy clarity:** transparent deterministic rules (no trained classifier).
- **Leakage gate** (`quality/leakage_gate.py` + tests): fails if any future model's numeric
  features reconstruct its label ≥ 0.95 — prevents the v2 mistake from recurring.

## Recommendations
1. **Risk (done):** unsupervised price-anomaly + transparent rules now shipped. Next: pursue a
   genuinely human-labelled trust/fraud dataset before any *supervised* risk model.
2. **Fake review:** deploy calibrated LinearSVC (TF-IDF) — best accuracy/cost; add isotonic
   calibration; guard against very short inputs (the truncation fragility).
3. **Sentiment:** expect domain-shift degradation; calibrate per-domain or add target-domain data.
4. **Production:** wire the experiment registry + model cards into the backend `/model-info`;
   add extraction validation + coverage metrics from the Phase 5 proposal.
5. **DL:** a GPU fine-tune of DistilBERT remains future work (CPU-infeasible here); the frozen
   embedding result suggests it is unlikely to beat TF-IDF for generated-vs-original detection.

## How to reproduce
See `reproducibility_report.md`. One command: `PYTHONPATH=. python ml/training/v3/run_v3.py`.
