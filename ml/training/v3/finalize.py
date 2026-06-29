"""Finalize the production model set: confirm each of the 5 signals loads and scores together.

This does NOT retrain — fake-review (v3 winner) and risk (v3 leakage-free) were trained by the
v3 pipeline; sentiment uses the v2 multi-class model. finalize.py verifies the chosen artifacts
load, scores a demo product end-to-end, and writes ml/reports/v3/PRODUCTION_MODELS.md. If a
chosen artifact is missing it raises, so this doubles as a finalize gate.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib

try:
    from ml.training.common import clean_text, write_json
    from ml.training.v3 import ARTIFACTS_DIR, REPORTS_DIR
    from ml.training.v3.risk_v3 import policy_clarity_score, price_safety_score, seller_reliability_score
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from training.common import clean_text, write_json  # type: ignore
    from training.v3 import ARTIFACTS_DIR, REPORTS_DIR  # type: ignore
    from training.v3.risk_v3 import policy_clarity_score, price_safety_score, seller_reliability_score  # type: ignore


REPO = Path(__file__).resolve().parents[2]

# Chosen production model per signal (paths span v2 + v3 — documented, not duplicated).
SELECTION = [
    {"signal": "review_authenticity", "model": "calibrated LinearSVC + word/char TF-IDF",
     "version": "v3", "artifact": "artifacts/v3/fake_review/model.joblib",
     "metric": "ROC-AUC 0.989 / acc 0.949 (in-domain), 0.91 cross-category", "leakage_safe": True},
    {"signal": "sentiment", "model": "TF-IDF + LogReg (neg/neu/pos)",
     "version": "v2", "artifact": "artifacts/v2/sentiment/sentiment_tfidf_logreg.joblib",
     "metric": "acc 0.867 (weak star labels)", "leakage_safe": True},
    {"signal": "seller_reliability", "model": "transparent rule (rating/reviews/tenure)",
     "version": "v3", "artifact": "(rule, no artifact)", "metric": "deterministic", "leakage_safe": True},
    {"signal": "price_safety", "model": "unsupervised IsolationForest anomaly + ratio rule",
     "version": "v3", "artifact": "artifacts/v3/risk/price_anomaly_iforest.joblib",
     "metric": "injected-anomaly recall 1.0", "leakage_safe": True},
    {"signal": "return_policy_clarity", "model": "transparent rule (return/period/warranty)",
     "version": "v3", "artifact": "(rule, no artifact)", "metric": "deterministic", "leakage_safe": True},
]

# v2 TrustScore weights (unchanged).
WEIGHTS = {"review_authenticity": 0.30, "seller_reliability": 0.20, "sentiment": 0.20,
           "return_policy_clarity": 0.15, "price_safety": 0.10, "user_feedback_history": 0.05}

_DEMO = {
    "title": "Wireless Earbuds Pro", "price": 8.0, "category_median": 60.0,
    "seller_rating": 2.6, "seller_reviews": 3, "seller_years": None,
    "return_policy": "All sales final.",
    "reviews": ["amazing perfect best ever", "amazing perfect best ever", "great"],
}


def _load_fake() -> tuple[Any, Any]:
    base = ARTIFACTS_DIR / "fake_review"
    model = joblib.load(base / "model.joblib")
    vec = joblib.load(base / "vectorizer.joblib")
    return model, vec


def _load_sentiment() -> Any:
    return joblib.load(REPO / "artifacts" / "v2" / "sentiment" / "sentiment_tfidf_logreg.joblib")


def _score_demo() -> dict[str, int]:
    fake_model, fake_vec = _load_fake()
    texts = [clean_text(r) for r in _DEMO["reviews"]]
    fake_prob = fake_model.predict_proba(fake_vec.transform(texts))[:, -1].mean()
    review_authenticity = round(100 * (1 - float(fake_prob)))

    senti = _load_sentiment()
    classes = [str(c) for c in senti.classes_]
    label_score = {"negative": 20, "neutral": 55, "positive": 90}
    probs = senti.predict_proba(texts)
    per = [sum(label_score.get(c, 50) * float(p) for c, p in zip(classes, row)) for row in probs]
    sentiment = round(sum(per) / len(per))

    seller = seller_reliability_score(_DEMO["seller_rating"], _DEMO["seller_reviews"], _DEMO["seller_years"])
    price = price_safety_score(_DEMO["price"], _DEMO["category_median"])
    policy = policy_clarity_score(_DEMO["return_policy"])
    return {"review_authenticity": review_authenticity, "sentiment": sentiment,
            "seller_reliability": seller, "price_safety": price, "return_policy_clarity": policy,
            "user_feedback_history": 50}


def run(args: argparse.Namespace) -> dict[str, Any]:
    # finalize gate: every chosen artifact must exist
    missing = [s["artifact"] for s in SELECTION
               if s["artifact"].startswith("artifacts/") and not (REPO / s["artifact"]).exists()]
    if missing:
        raise FileNotFoundError(f"Final artifacts missing: {missing}. Run the v3 pipeline first.")

    components = _score_demo()
    trust = round(sum(components[k] * w for k, w in WEIGHTS.items() if k in components))
    risk_level = "Low Risk" if trust >= 80 else "Medium Risk" if trust >= 50 else "High Risk"
    result = {"selection": SELECTION, "weights": WEIGHTS,
              "demo_product": {k: _DEMO[k] for k in ("title", "price", "category_median")},
              "demo_component_scores": components, "demo_trust_score": trust, "demo_risk_level": risk_level}
    write_json(REPORTS_DIR / "production_models.json", result)
    _write_md(result)
    print(f"[finalize] all chosen artifacts load + score. Demo TrustScore={trust} ({risk_level}).")
    print(f"[finalize] -> {REPORTS_DIR / 'PRODUCTION_MODELS.md'}")
    return result


def _write_md(r: dict[str, Any]) -> None:
    L = ["# TrustScore — Final Production Model Set", "",
         "The chosen model for each of the five signals, after v1→v2→v3. Paths span v2 and v3 "
         "(no duplication). All are leakage-safe.", "",
         "| Signal | Final model | Ver | Metric | Artifact |", "|---|---|---|---|---|"]
    for s in r["selection"]:
        L.append(f"| {s['signal']} | {s['model']} | {s['version']} | {s['metric']} | `{s['artifact']}` |")
    c = r["demo_component_scores"]
    L += ["", "## End-to-end finalize check (demo product: $8 vs $60 market, weak seller, no returns)", "",
          "All chosen artifacts loaded and scored together:", "",
          "| Component | Score |", "|---|---|",
          *[f"| {k} | {v} |" for k, v in c.items()],
          "", f"**TrustScore = {r['demo_trust_score']} → {r['demo_risk_level']}** "
          f"(weighted: {', '.join(f'{k} {w}' for k, w in r['weights'].items())}).", "",
          "## Verdict", "",
          "No retraining required — fake-review (v3 winner) and risk (v3 leakage-free) are freshly "
          "trained by the v3 pipeline; sentiment uses the validated v2 model; seller/policy are "
          "transparent rules. The finalize gate (`finalize.py`) fails if any chosen artifact is missing."]
    (REPORTS_DIR / "PRODUCTION_MODELS.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description=__doc__)


if __name__ == "__main__":
    run(build_parser().parse_args())
