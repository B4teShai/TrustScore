# v3 Reproducibility Report

## Environment
- Platform: macOS (Apple Silicon), CPU-only (no GPU).
- Python 3.13.9; virtualenv at `.venv/`.
- Key packages: scikit-learn 1.9.0, pandas 3.0.3, numpy, datasets 5.0.0,
  transformers 5.12.1, torch 2.12.1, sentence-transformers 5.6.0,
  xgboost 3.3.0, lightgbm 4.6.0, catboost 1.2.10, shap 0.52.0, matplotlib.
- Global seed: **42** (`ml/training/v3/__init__.py:SEED`).

## Determinism notes
- All splits, samplers and models use `random_state=42` / `seed=42`.
- Datasets are content-hashed (`dataset_versions.json`) so a re-run on the same data is verifiable.
- Embeddings use a frozen `sentence-transformers/all-MiniLM-L6-v2` (inference only, no training),
  so embedding outputs are deterministic per input.
- Experiment records append to `experiments.jsonl` with params, metrics and dataset hashes.

## Reproduce
```bash
# 1. install (already in .venv)
.venv/bin/python -m pip install xgboost lightgbm catboost sentence-transformers shap

# 2. acquire datasets (HF + GitHub)
PYTHONPATH=. .venv/bin/python ml/training/v3/data_sources/acquire.py

# 3. run the full v3 pipeline (each phase is also independently runnable)
PYTHONPATH=. .venv/bin/python ml/training/v3/run_v3.py
```

Individual phases:
```bash
PYTHONPATH=. .venv/bin/python ml/training/v3/data_sources/dataset_report_v3.py   # phase 1
PYTHONPATH=. .venv/bin/python ml/training/v3/quality/data_quality_audit.py        # phase 2
PYTHONPATH=. .venv/bin/python ml/training/v3/quality/leakage_analysis.py          # phase 2 (centerpiece)
PYTHONPATH=. .venv/bin/python ml/training/v3/feature_analysis_v3.py               # phase 3
PYTHONPATH=. .venv/bin/python ml/training/v3/leaderboard.py                       # phase 4
PYTHONPATH=. .venv/bin/python ml/training/v3/evaluation_v3.py                     # phase 6
PYTHONPATH=. .venv/bin/python ml/training/v3/extraction_coverage.py              # phase 5 scaffolding
```

## Outputs
- Models: `ml/artifacts/v3/` (+ `model_cards/`). Reports: `ml/reports/v3/`.
- v1 (`ml/artifacts/`) and v2 (`ml/artifacts/v2/`) are never written by v3.

## Tests
```bash
PYTHONPATH=. .venv/bin/python -m pytest ml/tests        # v1 + v2 + v3 smoke tests
```
