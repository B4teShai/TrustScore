"""Smoke tests for the v3 research package (fast, offline — no embeddings/HF).

Covers the pure-logic pieces: registry hashing/cards, linguistic + price features, the
leakage-reconstruction helper, dataset scoring and evaluation utilities.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ml.training.v3 import features_v3, finalize, risk_v3
from ml.training.v3.data_sources import dataset_report_v3
from ml.training.v3.evaluation_v3 import _drop_some_vowels, _ece
from ml.training.v3.quality.leakage_analysis import _numeric_leakage
from ml.training.v3.quality.leakage_gate import LeakageError, assert_leakage_free, is_leakage_free
from ml.training.v3.registry import content_hash, write_model_card


def test_content_hash_is_stable_and_path_independent(tmp_path: Path) -> None:
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    a.write_text("text,label\nhello,1\n", encoding="utf-8")
    b.write_text("text,label\nhello,1\n", encoding="utf-8")
    assert content_hash(a) == content_hash(b)
    assert content_hash(tmp_path / "missing.csv") == "missing"


def test_model_card_written(tmp_path: Path) -> None:
    p = write_model_card(model_id="t", task="x", intended_use="u", training_data="d",
                         metrics={"acc": 1.0}, limitations=["l"], ethical_notes=["e"], out_dir=tmp_path)
    assert p.exists() and "Model Card" in p.read_text()


def test_linguistic_features_shape_and_values() -> None:
    s = pd.Series(["GREAT product!!!", "ok"])
    feats = features_v3.linguistic_features(s)
    assert {"char_len", "type_token_ratio", "caps_ratio", "exclaim_count"} <= set(feats.columns)
    assert feats.loc[0, "exclaim_count"] == 3.0
    assert feats.loc[0, "caps_ratio"] > 0


def test_price_features_produce_anomaly_column() -> None:
    rng = np.random.default_rng(0)
    prices = list(rng.normal(50, 5, 80)) + [999.0, 0.5]  # two anomalies
    meta = pd.DataFrame({"price": prices, "main_category": ["All_Beauty"] * len(prices)})
    feats = features_v3.price_features(meta)
    assert {"price_cat_z", "price_log", "price_anomaly"} <= set(feats.columns)
    assert feats["price_anomaly"].notna().all()


def test_leakage_reconstruction_is_near_perfect_for_rule_label() -> None:
    # label is a deterministic function of the numeric features -> tree reconstructs it
    rng = np.random.default_rng(0)
    rating = rng.uniform(1, 5, 400)
    nrev = rng.integers(0, 500, 400).astype(float)
    df = pd.DataFrame({"rating": rating, "log_review_count": np.log1p(nrev)})
    df["seller_label"] = np.where((rating >= 4.2) & (nrev >= 50), "reliable",
                                  np.where((rating < 3.5) | (nrev < 5), "weak", "mixed"))
    res = _numeric_leakage(df, ["rating", "log_review_count"], "seller_label")
    assert res["label_reconstruction_accuracy"] >= 0.95


def test_eval_utilities() -> None:
    y = np.array([0, 1, 1, 0])
    prob = np.array([0.1, 0.9, 0.8, 0.2])
    assert 0.0 <= _ece(y, prob) <= 1.0
    assert _drop_some_vowels("aeiouaeiouaeiou") != ""


def test_dataset_report_scoring_helpers() -> None:
    assert dataset_report_v3._balance_score({"a": 50, "b": 50}) == 1.0
    assert dataset_report_v3._openness("Apache-2.0") >= 0.9
    e = dataset_report_v3._entry("k", "fake_review", 100000, "permissive", "d", {"0": 50, "1": 50}, "src", "n")
    assert 0.0 <= e["quality_score"] <= 1.0


def test_leakage_gate_flags_circular_label_and_passes_independent_one() -> None:
    rng = np.random.default_rng(0)
    rating = rng.uniform(1, 5, 400)
    nrev = rng.integers(0, 500, 400).astype(float)
    X = pd.DataFrame({"rating": rating, "log_review_count": np.log1p(nrev)})
    # circular label (function of X) -> gate must flag leakage
    circular = np.where((rating >= 4.2) & (nrev >= 50), 1, 0)
    assert not is_leakage_free(X, circular)
    try:
        assert_leakage_free(X, circular, name="circular")
        raise AssertionError("expected LeakageError")
    except LeakageError:
        pass
    # independent label (random) -> gate must pass
    independent = rng.integers(0, 2, 400)
    assert is_leakage_free(X, independent)


def test_risk_v3_rule_scores_are_monotonic_and_bounded() -> None:
    assert risk_v3.seller_reliability_score(4.8, 2000, 6) > risk_v3.seller_reliability_score(2.5, 3, None)
    assert risk_v3.price_safety_score(8, 60) < risk_v3.price_safety_score(55, 60)
    assert risk_v3.policy_clarity_score("30-day return and refund with warranty") > risk_v3.policy_clarity_score("all sales final")
    for fn, arg in [(risk_v3.seller_reliability_score, (None, None, None)), (risk_v3.price_safety_score, (None, None))]:
        assert 0 <= fn(*arg) <= 100


def test_finalize_selection_covers_all_signals_and_weights_sum_to_one() -> None:
    signals = {s["signal"] for s in finalize.SELECTION}
    assert {"review_authenticity", "sentiment", "seller_reliability", "price_safety", "return_policy_clarity"} <= signals
    assert all(s["leakage_safe"] for s in finalize.SELECTION)
    assert abs(sum(finalize.WEIGHTS.values()) - 1.0) < 1e-6
