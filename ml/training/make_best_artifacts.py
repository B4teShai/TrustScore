"""Assemble the per-signal *best* artifact set into ``ml/artifacts/best/``.

The repository keeps every trained model version under ``ml/artifacts/{v1,v2,v3}``
(version by version). This script gathers the single best leakage-safe model for
each of the five TrustScore signals into one curated directory, ``ml/artifacts/best/``,
plus a ``best_manifest.json`` describing why each was chosen.

It is idempotent and fast (a copy + manifest write), so it is safe to run on
container startup. Two signals (seller reliability, return-policy clarity) are
served by transparent deterministic rules and have no artifact by design.

Run:
    python -m ml.training.make_best_artifacts
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil


_REPO_ROOT = Path(__file__).resolve().parents[2]
_ARTIFACTS = _REPO_ROOT / "ml" / "artifacts"
_BEST = _ARTIFACTS / "best"


# (signal, source_dir, [files], dest_subdir, model description, metric, source version)
_SELECTION = [
    {
        "signal": "review_authenticity",
        "src": _ARTIFACTS / "v3" / "fake_review",
        "files": ["model.joblib", "vectorizer.joblib", "metadata.json"],
        "dest": "fake_review",
        "model": "calibrated LinearSVC + word/char TF-IDF",
        "metric": "ROC-AUC 0.989 / acc 0.949 in-domain, 0.91 cross-category",
        "version": "v3",
    },
    {
        "signal": "sentiment",
        "src": _ARTIFACTS / "v2" / "sentiment",
        "files": ["sentiment_tfidf_logreg.joblib", "sentiment_model_metadata.json"],
        "dest": "sentiment",
        "model": "TF-IDF + LogReg (neg/neu/pos)",
        "metric": "acc 0.867 (weak star labels)",
        "version": "v2",
    },
    {
        "signal": "price_safety",
        "src": _ARTIFACTS / "v3" / "risk",
        "files": [
            "price_anomaly_iforest.joblib",
            "price_anomaly_medians.json",
            "price_anomaly_metadata.json",
        ],
        "dest": "risk",
        "model": "unsupervised IsolationForest anomaly + ratio rule",
        "metric": "injected-anomaly recall 1.0",
        "version": "v3",
    },
]

_RULE_SIGNALS = [
    {
        "signal": "seller_reliability",
        "model": "transparent rule (rating / review volume / tenure)",
        "metric": "deterministic",
        "version": "v3",
    },
    {
        "signal": "return_policy_clarity",
        "model": "transparent rule (return / period / warranty wording)",
        "metric": "deterministic",
        "version": "v3",
    },
]


def assemble_best(*, verbose: bool = True) -> dict:
    """Copy each winning artifact into ml/artifacts/best/ and write the manifest."""
    _BEST.mkdir(parents=True, exist_ok=True)
    selection: list[dict] = []
    missing: list[str] = []

    for item in _SELECTION:
        src_dir: Path = item["src"]
        dest_dir = _BEST / item["dest"]
        dest_dir.mkdir(parents=True, exist_ok=True)
        copied: list[str] = []
        for name in item["files"]:
            src = src_dir / name
            if not src.is_file():
                missing.append(str(src.relative_to(_REPO_ROOT)))
                continue
            shutil.copy2(src, dest_dir / name)
            copied.append(name)
        selection.append(
            {
                "signal": item["signal"],
                "artifact": f"best/{item['dest']}/{item['files'][0]}",
                "files": copied,
                "model": item["model"],
                "metric": item["metric"],
                "source_version": item["version"],
                "leakage_safe": True,
            }
        )
        if verbose:
            print(f"[best] {item['signal']:<20} <- {item['version']}: {copied}")

    for rule in _RULE_SIGNALS:
        selection.append(
            {
                "signal": rule["signal"],
                "artifact": "(rule, no artifact)",
                "files": [],
                "model": rule["model"],
                "metric": rule["metric"],
                "source_version": rule["version"],
                "leakage_safe": True,
            }
        )
        if verbose:
            print(f"[best] {rule['signal']:<20} <- rule (no artifact)")

    manifest = {
        "name": "trustscore-best",
        "model_version": "1.0.0",
        "assembled_at": datetime.now(timezone.utc).isoformat(),
        "description": (
            "Curated best leakage-safe model per TrustScore signal. "
            "Version-by-version artifacts remain under ml/artifacts/{v1,v2,v3}."
        ),
        "weights": {
            "review_authenticity": 0.30,
            "seller_reliability": 0.20,
            "sentiment": 0.20,
            "return_policy_clarity": 0.15,
            "price_safety": 0.10,
            "user_feedback_history": 0.0,
        },
        "selection": selection,
        "missing_sources": missing,
    }
    (_BEST / "best_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    if verbose:
        if missing:
            print(f"[best] WARNING missing sources: {missing}")
        print(f"[best] manifest -> {(_BEST / 'best_manifest.json').relative_to(_REPO_ROOT)}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble ml/artifacts/best/")
    parser.add_argument("--quiet", action="store_true", help="suppress progress output")
    args = parser.parse_args()
    manifest = assemble_best(verbose=not args.quiet)
    if manifest["missing_sources"]:
        raise SystemExit(
            "Some best-artifact sources were missing; train the missing versions first."
        )


if __name__ == "__main__":
    main()
