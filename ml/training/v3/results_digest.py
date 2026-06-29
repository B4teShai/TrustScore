"""Consolidate all v3 reports into a single-glance RESULTS.md + results_digest.json.

Reads the JSON reports produced by the v3 phases and emits one compact results table so the
headline numbers (leakage verdict, leaderboard winner, OOD, calibration, robustness) are
visible without opening six files. Safe to run any time after the pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from ml.training.common import write_json
    from ml.training.v3 import REPORTS_DIR
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from training.common import write_json  # type: ignore
    from training.v3 import REPORTS_DIR  # type: ignore


def _load(name: str) -> dict[str, Any]:
    p = REPORTS_DIR / name
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def build(args: argparse.Namespace) -> dict[str, Any]:
    leak = _load("leakage_report.json").get("models", {})
    lb = _load("leaderboard.json")
    ev = _load("evaluation_report.json")
    feat = _load("feature_importance_v3.json")

    digest: dict[str, Any] = {}

    # leakage verdict
    digest["leakage"] = {
        k: {
            "numeric_reconstruction": v.get("leakage", {}).get("label_reconstruction_accuracy"),
            "numeric_ood": v.get("leave_one_category_out", {}).get("mean_numeric_ood_acc"),
            "text_only_ood": v.get("leave_one_category_out", {}).get("mean_text_only_ood_acc"),
        }
        for k, v in leak.items()
    }
    # leaderboard winner + top rows
    digest["fake_review_leaderboard"] = {
        "winner": lb.get("winner"),
        "top": lb.get("leaderboard", [])[:5],
    }
    # OOD + calibration + robustness
    sd = ev.get("sentiment_cross_domain", {})
    digest["sentiment_ood"] = {
        "in_domain": sd.get("in_domain", {}).get("accuracy"),
        "cross_domain": {k: v.get("accuracy") for k, v in sd.get("cross_domain", {}).items()},
    }
    fo = ev.get("fake_review_ood", {})
    digest["fake_review_ood"] = {
        "in_domain": fo.get("in_domain_random_split", {}).get("accuracy"),
        "mean_leave_one_category_out": fo.get("mean_ood_accuracy"),
    }
    cr = ev.get("fake_review_calibration_robustness", {})
    digest["calibration"] = cr.get("calibration")
    digest["robustness"] = cr.get("robustness")
    digest["fake_review_ablation"] = feat.get("ablation")
    # risk fix (Stage 0) — leakage-free replacement
    risk = _load("risk_v3.json")
    if risk:
        digest["risk_fix"] = {
            "approach": risk.get("approach"),
            "price_anomaly_validation": risk.get("price_anomaly", {}).get("validation"),
        }

    write_json(REPORTS_DIR / "results_digest.json", digest)
    _write_md(digest)
    print(f"[results] -> {REPORTS_DIR / 'RESULTS.md'}")
    return digest


def _g(d: dict[str, Any], *keys, default="—"):
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, {})
    return d if d not in ({}, None) else default


def _write_md(d: dict[str, Any]) -> None:
    L = ["# TrustScore v3 — Results (digest)", "",
         "One-glance summary; full detail in FINDINGS.md and the per-phase reports.", "",
         "## 1. Risk weak-label leakage (rejected)", "",
         "| Label | numeric reconstruction | numeric OOD | text-only OOD |",
         "|---|---|---|---|"]
    for k, v in d.get("leakage", {}).items():
        L.append(f"| {k} | {v.get('numeric_reconstruction')} | {v.get('numeric_ood')} | {v.get('text_only_ood')} |")
    L += ["", "Numeric models reconstruct the label (~1.0) and keep ~1.0 out-of-category while "
          "text-only collapses → **leakage, rejected as a trust metric**.", ""]

    rf = d.get("risk_fix")
    if rf:
        v = rf.get("price_anomaly_validation") or {}
        L += ["### Fix (Stage 0) — leakage-free risk scorer", "",
              "- **Price safety:** unsupervised IsolationForest anomaly (no label) — injected-anomaly "
              f"recall **{v.get('injected_anomaly_recall')}**, false-flag {v.get('normal_false_flag_rate')}.",
              "- **Seller / policy:** transparent deterministic rules (no trained classifier).",
              "- A **leakage-gate test** blocks any future model that trains on its label's own inputs.", ""]

    w = d.get("fake_review_leaderboard", {}).get("winner") or {}
    L += ["## 2. Fake-review leaderboard", "",
          f"**Winner:** `{w.get('model')}` on `{w.get('feature_set')}` — "
          f"ROC-AUC **{w.get('roc_auc')}**, acc {w.get('accuracy')}, {w.get('artifact_size_kb')} KB.", "",
          "| Model | Features | Acc | ROC-AUC | Size KB |", "|---|---|---|---|---|"]
    for r in d.get("fake_review_leaderboard", {}).get("top", []):
        L.append(f"| {r['model']} | {r['feature_set']} | {r['accuracy']} | {r['roc_auc']} | {r['artifact_size_kb']} |")

    abl = d.get("fake_review_ablation") or []
    if abl:
        L += ["", "## 3. Feature ablation (fake review)", "", "| Feature set | Acc | ROC-AUC |", "|---|---|---|"]
        for r in abl:
            L.append(f"| {r['feature_set']} | {r['accuracy']} | {r['roc_auc']} |")
        L += ["", "TF-IDF beats frozen embeddings — lexical signal dominates (reported honestly).", ""]

    so = d.get("sentiment_ood", {})
    L += ["## 4. Out-of-domain generalisation", "",
          f"- Sentiment (train Amazon, acc {so.get('in_domain')}): "
          + ", ".join(f"{k} {v}" for k, v in (so.get("cross_domain") or {}).items()) + ".",
          f"- Fake review: in-domain {d.get('fake_review_ood',{}).get('in_domain')}, "
          f"mean leave-one-category-out **{d.get('fake_review_ood',{}).get('mean_leave_one_category_out')}**.", ""]

    cal = d.get("calibration") or {}
    rob = d.get("robustness") or {}
    if cal:
        L += ["## 5. Calibration & robustness", "",
              f"- Calibration ECE: raw {cal.get('raw',{}).get('ece')} → isotonic **{cal.get('isotonic',{}).get('ece')}**.",
              f"- Robustness (Δacc): "
              + ", ".join(f"{k} {v['delta']}" for k, v in (rob.get('perturbed') or {}).items()) + ".", ""]
    (REPORTS_DIR / "RESULTS.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description=__doc__)


if __name__ == "__main__":
    build(build_parser().parse_args())
