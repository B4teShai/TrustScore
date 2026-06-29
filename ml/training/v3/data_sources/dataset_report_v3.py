"""Phase 1 — dataset comparison, quality ranking, licensing review, recommendation.

Reads the acquisition manifest plus inherited v1/v2 datasets, scores each on size, class
balance, license openness and domain relevance, ranks them, and writes
ml/reports/v3/dataset_comparison.{json,md}. Gated/manual datasets are documented, not scored.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from ml.training.common import write_json
    from ml.training.v3 import DATA_DIR, REPORTS_DIR
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[3]))
    from training.common import write_json  # type: ignore
    from training.v3 import DATA_DIR, REPORTS_DIR  # type: ignore


# License openness score (0-1) for ranking transparency.
LICENSE_OPENNESS = {
    "permissive": 1.0, "apache": 0.9, "research": 0.6, "terms": 0.4, "noncommercial": 0.5, "unknown": 0.3,
}


def _openness(license_str: str) -> float:
    s = license_str.lower()
    for key, val in LICENSE_OPENNESS.items():
        if key in s:
            return val
    if "cc-by" in s:
        return 0.7
    return 0.4


def _balance_score(dist: dict[str, int]) -> float:
    if not dist:
        return 0.0
    vals = list(dist.values())
    return round(min(vals) / max(vals), 3) if max(vals) else 0.0


def _entry(key: str, task: str, rows: int, license_str: str, domain: str,
           dist: dict[str, int], source: str, notes: str) -> dict[str, Any]:
    balance = _balance_score(dist)
    size_score = min(1.0, rows / 100_000)
    openness = _openness(license_str)
    quality = round(0.4 * balance + 0.3 * size_score + 0.3 * openness, 3)
    return {"key": key, "task": task, "rows": rows, "license": license_str, "license_openness": openness,
            "domain": domain, "class_balance": balance, "size_score": round(size_score, 3),
            "quality_score": quality, "source": source, "label_distribution": dist, "notes": notes}


def build(args: argparse.Namespace) -> dict[str, Any]:
    manifest = json.loads((DATA_DIR / "v3_dataset_manifest.json").read_text())
    entries: list[dict[str, Any]] = []

    # acquired (HF, curated)
    for a in manifest.get("acquired", []):
        entries.append(_entry(a["key"], a["task"], a["rows"], a["license"], a["domain"],
                              a.get("label_distribution", {}), a["source_id"], a.get("notes", "")))

    # second fake-review source (cross-category OOD)
    cross = DATA_DIR / "fake_review_cross.csv"
    if cross.exists():
        df = pd.read_csv(cross)
        dist = {str(k): int(v) for k, v in df["label"].value_counts().to_dict().items()}
        entries.append(_entry("fake_review_cross", "fake_review", int(len(df)),
                              "Research use (Salminen et al. fake reviews)", "amazon product reviews (multi-category)",
                              dist, "astrosbd/fake-review", "Real vs computer-generated; category column enables OOD holdout."))

    # inherited v1/v2 datasets
    inherited = [
        ("fake_reviews_v2", "fake_review", DATA_DIR.parent / "fake_reviews.csv", "label",
         "Research use (theArijitDas)", "amazon reviews", "v1/v2 fake review training set."),
        ("amazon_reviews_v2", "sentiment", DATA_DIR.parent / "amazon_reviews_sample.csv", None,
         "Amazon review data terms", "amazon (3 categories)", "v2 multi-category sentiment set (weak star labels)."),
        ("amazon_meta_v2", "risk", DATA_DIR.parent / "amazon_meta_sample.csv", None,
         "Amazon review data terms", "amazon metadata", "v2 risk metadata; weak-label leakage — see leakage_report.md."),
    ]
    for key, task, path, label_col, lic, domain, notes in inherited:
        if not Path(path).exists():
            continue
        if label_col:
            col = pd.read_csv(path, low_memory=False, usecols=[label_col])
            rows = int(len(col))
            dist = {str(k): int(v) for k, v in col[label_col].value_counts().to_dict().items()}
        else:
            # count rows via a single column (robust to newlines inside quoted text fields)
            first_col = pd.read_csv(path, low_memory=False, usecols=[0])
            rows = int(len(first_col))
            dist = {}
        entries.append(_entry(key, task, rows, lic, domain, dist, str(path.name), notes))

    entries.sort(key=lambda e: e["quality_score"], reverse=True)
    for i, e in enumerate(entries, 1):
        e["rank"] = i

    report = {"datasets": entries, "gated_documented": manifest.get("gated", []),
              "unavailable": manifest.get("unavailable", []),
              "recommendation": _recommendation(entries)}
    write_json(REPORTS_DIR / "dataset_comparison.json", report)
    _write_md(report)
    print(f"[dataset-report] -> {REPORTS_DIR / 'dataset_comparison.md'}")
    return report


def _recommendation(entries: list[dict[str, Any]]) -> dict[str, str]:
    by_task: dict[str, list[dict[str, Any]]] = {}
    for e in entries:
        by_task.setdefault(e["task"], []).append(e)
    rec = {}
    for task, items in by_task.items():
        best = max(items, key=lambda e: e["quality_score"])
        rec[task] = f"{best['key']} (quality {best['quality_score']}, {best['rows']} rows, {best['license']})"
    return rec


def _write_md(report: dict[str, Any]) -> None:
    L = ["# v3 Dataset Comparison & Recommendation", "",
         "Quality score = 0.4·class-balance + 0.3·size + 0.3·license-openness.", "",
         "| Rank | Dataset | Task | Rows | Balance | License | Openness | Quality |",
         "|---|---|---|---|---|---|---|---|"]
    for e in report["datasets"]:
        L.append(f"| {e['rank']} | {e['key']} | {e['task']} | {e['rows']} | {e['class_balance']} | "
                 f"{e['license']} | {e['license_openness']} | **{e['quality_score']}** |")
    L += ["", "## Recommendation by task", ""]
    for task, rec in report["recommendation"].items():
        L.append(f"- **{task}:** {rec}")
    L += ["", "## Gated / manual (documented, not downloaded)", ""]
    for g in report["gated_documented"]:
        L.append(f"- **{g['key']}** — {g['reason']}")
    if report.get("unavailable"):
        L += ["", "## Unavailable (HF ids did not resolve)", ""]
        for u in report["unavailable"]:
            L.append(f"- {u['key']} ({u['task']}) — tried {u['tried']}")
    (REPORTS_DIR / "dataset_comparison.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description=__doc__)


if __name__ == "__main__":
    build(build_parser().parse_args())
