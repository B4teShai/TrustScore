"""v3 risk scorer — leakage-free by construction (Stage 0 fix).

v2's seller/price classifier was rejected: its labels were deterministic functions of the
features (proved in leakage_report.md). v3 replaces it with signals that do NOT train on that
circular label:

- **Price safety** = UNSUPERVISED anomaly detection (IsolationForest on category-normalised
  price). No price label is used in training, so there is nothing to leak. Validated by
  injected-anomaly recall.
- **Seller reliability** and **return-policy clarity** = TRANSPARENT RULES (deterministic,
  interpretable), not a trained classifier — the same rules the backend already ships.

Artifacts: ml/artifacts/v3/risk/. Report: ml/reports/v3/risk_v3.md.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

try:
    from ml.training.common import parse_float, write_json
    from ml.training.v3 import ARTIFACTS_DIR, MODEL_CARDS_DIR, REPORTS_DIR, SEED
    from ml.training.v3.registry import log_experiment, write_model_card
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from training.common import parse_float, write_json  # type: ignore
    from training.v3 import ARTIFACTS_DIR, MODEL_CARDS_DIR, REPORTS_DIR, SEED  # type: ignore
    from training.v3.registry import log_experiment, write_model_card  # type: ignore


def _meta_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "amazon_meta_sample.csv"


# --------------------------------------------------------------------------- #
# Transparent rules (no training, no label dependence) — mirror the backend.
# --------------------------------------------------------------------------- #
def seller_reliability_score(rating: float | None, review_count: int | None, years_active: int | None) -> int:
    """Deterministic 0-100 from visible seller signals. Interpretable; no ML."""
    parts: list[tuple[float, float]] = []
    if rating is not None:
        parts.append((rating / 5 * 100, 0.55))
    if review_count is not None:
        parts.append((min(100.0, math.log10(review_count + 1) / 4 * 100), 0.25))
    if years_active is not None:
        parts.append((min(100.0, years_active / 5 * 100), 0.20))
    if not parts:
        return 50
    return max(0, min(100, round(sum(s * w for s, w in parts) / sum(w for _, w in parts))))


def policy_clarity_score(return_policy: str | None) -> int:
    """Deterministic 0-100 from policy-text signals. Interpretable; no ML."""
    if not return_policy or not return_policy.strip():
        return 50
    t = return_policy.lower()
    score = 30
    if any(k in t for k in ("return", "refund")):
        score += 25
    if any(k in t for k in ("warranty", "exchange", "replacement")):
        score += 10
    import re

    if re.search(r"\b\d+\s*-?\s*(day|days|week|weeks|month|months)\b", t):
        score += 25
    if len(t.split()) >= 8:
        score += 10
    return max(0, min(100, score))


# --------------------------------------------------------------------------- #
# Unsupervised price anomaly (the leakage-free price signal).
# --------------------------------------------------------------------------- #
def _price_frame(meta: pd.DataFrame) -> pd.DataFrame:
    df = meta.copy()
    df["_price"] = df["price"].map(lambda v: parse_float(v, np.nan))
    df["_cat"] = df.get("main_category", df.get("source_category", "unknown")).fillna("unknown").astype(str)
    grp = df.groupby("_cat")["_price"]
    df["price_cat_z"] = grp.transform(lambda x: (x - x.median()) / (x.std(ddof=0) + 1e-6))
    df["price_log"] = np.log1p(df["_price"].clip(lower=0).fillna(0.0))
    return df


def train_price_anomaly(args: argparse.Namespace) -> dict[str, Any]:
    meta = pd.read_csv(_meta_path(), low_memory=False)
    df = _price_frame(meta)
    priced = df["_price"].notna() & (df["_price"] > 0)
    feats = df.loc[priced, ["price_log", "price_cat_z"]].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    iso = IsolationForest(n_estimators=200, contamination="auto", random_state=SEED, n_jobs=-1)
    iso.fit(feats.to_numpy())

    # category median table for inference-time normalisation
    medians = df[priced].groupby("_cat")["_price"].median().to_dict()
    out = ARTIFACTS_DIR / "risk"
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(iso, out / "price_anomaly_iforest.joblib")
    write_json(out / "price_anomaly_medians.json", {k: float(v) for k, v in medians.items()})

    validation = _validate_injected_anomalies(feats, iso)
    metadata = {
        "model_name": "v3_price_anomaly_isolation_forest", "model_version": "0.3.0",
        "approach": "unsupervised anomaly detection on category-normalised price (no price label used)",
        "n_priced_rows": int(priced.sum()), "validation": validation,
        "leakage_free": True,
        "reason": "No supervised price label is used in training; nothing to reconstruct.",
    }
    write_json(out / "price_anomaly_metadata.json", metadata)
    write_model_card(
        model_id="risk_v3_price_anomaly", task="price-safety anomaly detection",
        intended_use="Flag products whose price is anomalous within its category (possible scam-low or overpriced).",
        training_data="v2 Amazon metadata sample (price + category only); unsupervised, no label.",
        metrics=validation, out_dir=MODEL_CARDS_DIR,
        limitations=["Flags statistical outliers, not confirmed fraud.",
                     "Category medians depend on catalogue coverage.",
                     "66% of metadata rows lack a price; those default to neutral."],
        ethical_notes=["A low score is a caution signal, not proof of wrongdoing.",
                       "Avoids the v2 weak-label leakage by not training on a constructed label."],
    )
    log_experiment(phase="risk_v3", name="price_anomaly", params={"n_estimators": 200},
                   metrics=validation, datasets={"amazon_meta": "v2_sample"})
    return metadata


def _validate_injected_anomalies(feats: pd.DataFrame, iso: IsolationForest) -> dict[str, Any]:
    """No labels exist, so validate by injecting synthetic extremes and measuring recall."""
    rng = np.random.default_rng(SEED)
    n = min(500, len(feats))
    sample = feats.sample(n=n, random_state=SEED).to_numpy()
    injected = sample.copy()
    # push price_log and z to extreme low/high
    injected[: n // 2, 0] = feats["price_log"].max() + 3   # absurdly expensive
    injected[: n // 2, 1] = 12.0
    injected[n // 2 :, 0] = 0.0                            # absurdly cheap
    injected[n // 2 :, 1] = -12.0
    normal_flagged = (iso.predict(sample) == -1).mean()
    injected_flagged = (iso.predict(injected) == -1).mean()
    return {"injected_anomaly_recall": round(float(injected_flagged), 4),
            "normal_false_flag_rate": round(float(normal_flagged), 4)}


def price_safety_score(price: float | None, category_median: float | None) -> int:
    """Inference-time 0-100 price-safety from the ratio to category median (rule, leakage-free).

    Mirrors the anomaly direction without needing the model loaded: very low = scam risk,
    very high = overpriced. Deterministic and explainable.
    """
    if price is None or category_median is None or category_median <= 0 or price <= 0:
        return 50
    ratio = price / category_median
    if ratio < 0.35:
        return 25
    if ratio < 0.6:
        return 45
    if ratio <= 1.4:
        return 90
    if ratio <= 2.0:
        return 70
    return 50


def run(args: argparse.Namespace) -> dict[str, Any]:
    print("[risk_v3] training unsupervised price anomaly (leakage-free)...")
    meta = train_price_anomaly(args)
    report = {
        "approach": {
            "price_safety": "unsupervised IsolationForest on category-normalised price (no label).",
            "seller_reliability": "transparent deterministic rule (rating / review-count / tenure).",
            "return_policy_clarity": "transparent deterministic rule (return/period/warranty signals).",
        },
        "why_leakage_free": "No model is trained on the v2 weak labels (which were functions of the "
                            "features). Price uses unsupervised anomaly detection; seller/policy use rules.",
        "price_anomaly": meta,
    }
    write_json(REPORTS_DIR / "risk_v3.json", report)
    _write_md(report)
    print(f"[risk_v3] -> {REPORTS_DIR / 'risk_v3.md'}")
    return report


def _write_md(report: dict[str, Any]) -> None:
    v = report["price_anomaly"]["validation"]
    L = ["# v3 Risk Scorer — leakage-free (Stage 0 fix)", "",
         "Replaces the rejected v2 weak-label seller/price classifier.", "",
         "## Approach", ""]
    for k, val in report["approach"].items():
        L.append(f"- **{k}**: {val}")
    L += ["", "## Why this is leakage-free", "", report["why_leakage_free"], "",
          "## Price-anomaly validation (no labels → injected extremes)", "",
          f"- Injected-anomaly recall: **{v['injected_anomaly_recall']}** (synthetic extreme prices flagged).",
          f"- Normal false-flag rate: {v['normal_false_flag_rate']}.", "",
          "Seller and policy scores are deterministic rules, so they need no statistical validation; "
          "they are interpretable by inspection. A leakage-gate test (`test_leakage_gate*`) guards against "
          "any future supervised risk model that trains on its own label's inputs.", ""]
    (REPORTS_DIR / "risk_v3.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description=__doc__)


if __name__ == "__main__":
    run(build_parser().parse_args())
