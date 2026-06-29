"""Feature engineering shared by the v2 trainers.

The risk models are the headline change. In v1 the seller/price/policy labels are
deterministic functions of *numeric* metadata (rating, review count, price ratio) and
text-derived flags, yet v1 fed the models *only* product title text through TF-IDF, so
the signal that defines the label was never available. v2 builds a ``ColumnTransformer``
that combines those numeric features with the text.

Every numeric feature here is reconstructable by the FastAPI backend from fields it
already extracts (seller rating, review count, price, average market price, return
policy text), so the trained pipelines stay drop-in for inference. ``feature_spec.json``
written next to each artifact tells the backend which columns to populate.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from ml.training.common import clean_text, parse_float
except ModuleNotFoundError:  # pragma: no cover - exercised when run as a script
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from common import clean_text, parse_float  # type: ignore[no-redef]


_PERIOD_RE = re.compile(r"\b\d+\s*-?\s*(day|days|week|weeks|month|months)\b")


# --------------------------------------------------------------------------- #
# Numeric derivations (mirrored in apps/api/app/ml/risk_service.py for the v2 path)
# --------------------------------------------------------------------------- #
def log_review_count(value: object) -> float:
    """log1p of the rating/review count; stable and monotonic for the seller label."""
    return float(np.log1p(max(parse_float(value, 0.0), 0.0)))


def price_ratio(price: object, reference: object) -> float:
    """price / reference (category median at train time, market price at inference)."""
    p = parse_float(price, 0.0)
    ref = parse_float(reference, 0.0)
    if p <= 0 or ref <= 0:
        return 1.0
    return float(p / ref)


def policy_flags(policy: object) -> dict[str, int]:
    """Boolean policy signals; the v1 policy label is a function of exactly these."""
    text = clean_text(policy)
    has_return = int(("return" in text) or ("refund" in text))
    has_period = int(bool(_PERIOD_RE.search(text)))
    has_warranty = int(
        ("warranty" in text) or ("exchange" in text) or ("replacement" in text)
    )
    return {"has_return": has_return, "has_period": has_period, "has_warranty": has_warranty}


# --------------------------------------------------------------------------- #
# Feature specs: the contract between training and the backend.
# numeric_features are ordered; text_feature is a single column. label_score_map is
# duplicated here for documentation only (the backend keeps its own authoritative maps).
# --------------------------------------------------------------------------- #
RISK_FEATURE_SPECS: dict[str, dict[str, Any]] = {
    "seller_reliability": {
        "numeric_features": ["rating", "log_review_count"],
        "text_feature": "seller_text",
        "label_score_map": {"reliable": 88, "mixed": 60, "weak": 32},
    },
    "price_safety": {
        "numeric_features": ["price", "price_ratio"],
        "text_feature": "price_text",
        "label_score_map": {"normal": 90, "high_price": 65, "suspicious_low": 35},
    },
    "return_policy_clarity": {
        "numeric_features": ["has_return", "has_period", "has_warranty"],
        "text_feature": "policy_text",
        "label_score_map": {"clear": 90, "partial": 65, "unclear": 35},
    },
}


def make_risk_pipeline(
    *,
    numeric_features: list[str],
    text_feature: str,
    classifier: Any,
    max_features: int,
    max_ngram: int,
    min_df: int,
) -> Pipeline:
    """Build a DataFrame-consuming ColumnTransformer + classifier pipeline.

    Inference contract: ``pipeline.predict_proba(df)`` / ``pipeline.predict(df)`` where
    ``df`` is a pandas DataFrame holding ``numeric_features`` + ``text_feature`` columns.
    ``pipeline.classes_`` exposes the string labels, matching the v1 backend interface.
    """
    transformer = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            (
                "txt",
                TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=(1, max_ngram),
                    min_df=min_df,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
                text_feature,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    return Pipeline([("features", transformer), ("classifier", classifier)])


def make_text_only_pipeline(
    *,
    text_feature: str,
    classifier: Any,
    max_features: int,
    max_ngram: int,
    min_df: int,
) -> Pipeline:
    """v1-style text-only pipeline, used for the ablation baseline."""
    transformer = ColumnTransformer(
        transformers=[
            (
                "txt",
                TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=(1, max_ngram),
                    min_df=min_df,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
                text_feature,
            )
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    return Pipeline([("features", transformer), ("classifier", classifier)])


def build_fake_review_vectorizer(
    *, max_features: int, word_ngram: int, char_min: int, char_max: int
) -> Any:
    """Word (1..n) TF-IDF unioned with char_wb (char_min..char_max) TF-IDF.

    Char n-grams catch the surface artifacts of machine-generated reviews that word
    n-grams miss. Returns a FeatureUnion that ``.fit_transform``/``.transform`` a list of
    strings, keeping the vectorizer/model split the backend expects.
    """
    from sklearn.pipeline import FeatureUnion

    word_vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, word_ngram),
        min_df=2,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(char_min, char_max),
        max_features=max_features // 2,
        min_df=2,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    return FeatureUnion([("word", word_vec), ("char", char_vec)])
