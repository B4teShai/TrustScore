"""TrustScore v2 training package.

Second-iteration training pipeline. v1 scripts and ``ml/artifacts/`` are left
untouched; v2 writes versioned artifacts to ``ml/artifacts/v2/`` (model_version 0.2.0).

Headline improvement: the weak-label risk models (seller / price / policy) are
upgraded from text-only TF-IDF to a ``ColumnTransformer`` that combines the numeric
metadata features the labels are actually derived from (rating, review count,
price ratio) with the product text. See ``train_risk_models_v2.py``.
"""
