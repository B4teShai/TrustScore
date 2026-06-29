"""Tests for the best-artifact assembler."""

import json
from pathlib import Path

from ml.training.make_best_artifacts import assemble_best


_SIGNALS = {
    "review_authenticity",
    "seller_reliability",
    "sentiment",
    "return_policy_clarity",
    "price_safety",
}


def test_assemble_best_covers_all_five_signals() -> None:
    manifest = assemble_best(verbose=False)
    signals = {entry["signal"] for entry in manifest["selection"]}
    assert _SIGNALS.issubset(signals)
    assert manifest["missing_sources"] == []


def test_best_dir_has_model_files_and_manifest() -> None:
    assemble_best(verbose=False)
    best = Path(__file__).resolve().parents[1] / "artifacts" / "best"
    assert (best / "best_manifest.json").is_file()
    assert (best / "fake_review" / "model.joblib").is_file()
    assert (best / "sentiment" / "sentiment_tfidf_logreg.joblib").is_file()
    assert (best / "risk" / "price_anomaly_iforest.joblib").is_file()

    manifest = json.loads((best / "best_manifest.json").read_text())
    rule_signals = {
        entry["signal"] for entry in manifest["selection"] if not entry["files"]
    }
    assert rule_signals == {"seller_reliability", "return_policy_clarity"}
