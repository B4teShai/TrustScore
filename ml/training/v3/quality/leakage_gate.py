"""Reusable leakage gate — prevents the v2 weak-label mistake from coming back.

Principle: never train a model on the numeric features that its label was *derived* from. This
gate trains a shallow tree on the candidate numeric features and, if it reconstructs the label
almost perfectly, flags leakage. Use it in tests/CI before trusting any supervised risk metric.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


class LeakageError(AssertionError):
    """Raised when numeric features reconstruct the label above the threshold."""


def numeric_reconstruction_score(
    X: pd.DataFrame | np.ndarray, y: Any, *, seed: int = 42, max_depth: int = 4
) -> float:
    """Accuracy of a shallow tree predicting the label from numeric features alone.

    ~1.0 means the label is a deterministic function of these features (leakage shortcut).
    """
    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=seed)
    tree.fit(X, y)
    return float(accuracy_score(y, tree.predict(X)))


def is_leakage_free(X: pd.DataFrame | np.ndarray, y: Any, *, threshold: float = 0.95, seed: int = 42) -> bool:
    """True when the label cannot be reconstructed from the numeric features (recon < threshold)."""
    return numeric_reconstruction_score(X, y, seed=seed) < threshold


def assert_leakage_free(
    X: pd.DataFrame | np.ndarray, y: Any, *, threshold: float = 0.95, seed: int = 42, name: str = "model"
) -> float:
    """Raise LeakageError if the numeric features reconstruct the label. Returns the score."""
    score = numeric_reconstruction_score(X, y, seed=seed)
    if score >= threshold:
        raise LeakageError(
            f"[{name}] numeric-only label reconstruction {score:.3f} >= {threshold}: "
            f"the label appears to be a function of these features (leakage). "
            f"Use features disjoint from the label's derivation, or an independent label."
        )
    return score
