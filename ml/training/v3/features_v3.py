"""v3 feature engineering.

Three honest building blocks:
- Review text: frozen sentence-transformer embeddings (no fine-tune) + linguistic features.
- Seller: category-normalised rating + within-category review-count percentile.
- Price: category-normalised z-score + UNSUPERVISED IsolationForest anomaly score.

The price/seller features here are computed without reference to the v2 weak labels, so the
unsupervised price anomaly scorer is the leakage-free risk reframe (see leakage_analysis.py).
Features the data cannot support (per-reviewer history, seller age/velocity time series) are
documented as gaps rather than fabricated.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

_WORD_RE = re.compile(r"\w+")
_DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=2)
def _load_embedder(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed_texts(
    texts: list[str], *, model_name: str = _DEFAULT_EMBED_MODEL, batch_size: int = 64
) -> np.ndarray:
    """Frozen embeddings (384-d for MiniLM). CPU-friendly, inference only."""
    model = _load_embedder(model_name)
    return np.asarray(
        model.encode(texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)
    )


def linguistic_features(texts: pd.Series) -> pd.DataFrame:
    """Cheap surface/style features that complement embeddings (vectorised)."""
    s = texts.fillna("").astype(str)
    char_len = s.str.len().clip(lower=1)
    words = s.map(lambda t: _WORD_RE.findall(t.lower()))
    word_len = words.map(len).clip(lower=1)
    uniq = words.map(lambda w: len(set(w)))
    caps = s.map(lambda t: sum(1 for c in t if c.isupper()))
    punct = s.map(lambda t: sum(1 for c in t if c in "!?.,;:"))
    excl = s.str.count("!")
    return pd.DataFrame(
        {
            "char_len": char_len.astype(float),
            "word_len": word_len.astype(float),
            "avg_word_len": (char_len / word_len).astype(float),
            "type_token_ratio": (uniq / word_len).astype(float),
            "caps_ratio": (caps / char_len).astype(float),
            "punct_ratio": (punct / char_len).astype(float),
            "exclaim_count": excl.astype(float),
        }
    )


def category_normalized_seller_features(meta: pd.DataFrame) -> pd.DataFrame:
    """Seller features that normalise rating/review-count within each category."""
    from ml.training.common import parse_float

    df = meta.copy()
    df["_rating"] = df["average_rating"].map(lambda v: parse_float(v, np.nan))
    df["_nrev"] = df["rating_number"].map(lambda v: parse_float(v, np.nan))
    cat = df.get("main_category", pd.Series(["unknown"] * len(df))).fillna("unknown").astype(str)
    df["_cat"] = cat
    grp = df.groupby("_cat")
    df["rating_cat_z"] = grp["_rating"].transform(lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-6))
    df["review_count_cat_pct"] = grp["_nrev"].transform(lambda x: x.rank(pct=True))
    df["log_review_count"] = np.log1p(df["_nrev"].fillna(0.0))
    return df[["rating_cat_z", "review_count_cat_pct", "log_review_count"]].fillna(0.0)


def price_features(meta: pd.DataFrame, *, seed: int = 42) -> pd.DataFrame:
    """Category-normalised price z-score + UNSUPERVISED IsolationForest anomaly score.

    Deliberately does not use the v2 price label — this is the leakage-free price signal.
    """
    from sklearn.ensemble import IsolationForest

    from ml.training.common import parse_float

    df = meta.copy()
    df["_price"] = df["price"].map(lambda v: parse_float(v, np.nan))
    cat = df.get("main_category", pd.Series(["unknown"] * len(df))).fillna("unknown").astype(str)
    df["_cat"] = cat
    grp = df.groupby("_cat")["_price"]
    df["price_cat_z"] = grp.transform(lambda x: (x - x.median()) / (x.std(ddof=0) + 1e-6))
    df["price_log"] = np.log1p(df["_price"].clip(lower=0).fillna(0.0))

    priced = df["_price"].notna() & (df["_price"] > 0)
    df["price_anomaly"] = 0.0
    if priced.sum() >= 50:
        feats = df.loc[priced, ["price_log", "price_cat_z"]].replace([np.inf, -np.inf], 0.0).fillna(0.0)
        iso = IsolationForest(n_estimators=150, contamination="auto", random_state=seed, n_jobs=-1)
        iso.fit(feats)
        # higher = more anomalous (flip sign of decision_function)
        df.loc[priced, "price_anomaly"] = -iso.decision_function(feats)
    return df[["price_cat_z", "price_log", "price_anomaly"]].replace([np.inf, -np.inf], 0.0).fillna(0.0)


# Documented feature gaps: the public metadata has no per-reviewer ids, no review timestamps
# at the seller level, and no historical snapshots, so reviewer-history, rating-velocity,
# seller-age and review-growth features cannot be computed from the available data.
FEATURE_GAPS: dict[str, str] = {
    "reviewer_history": "No per-reviewer id in the review datasets used.",
    "rating_velocity": "No time-series of ratings per seller/product in the metadata.",
    "seller_age": "No seller registration/first-seen date in the metadata.",
    "review_growth": "No longitudinal review-count snapshots available.",
}
