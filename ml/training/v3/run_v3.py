"""v3 orchestrator — runs the research phases in order and writes a summary.

Each phase is independently runnable and guarded so one failure does not abort the rest.
Assumes datasets are already acquired (run data_sources/acquire.py first, or pass --acquire).
"""

from __future__ import annotations

import argparse
import traceback
from typing import Any

try:
    from ml.training.common import write_json
    from ml.training.v3 import DATA_DIR, REPORTS_DIR
    from ml.training.v3.data_sources import acquire, dataset_report_v3
    from ml.training.v3 import evaluation_v3, feature_analysis_v3, finalize, leaderboard, results_digest, risk_v3
    from ml.training.v3.quality import data_quality_audit, leakage_analysis
    from ml.training.v3.registry import dataset_manifest
except ModuleNotFoundError:  # pragma: no cover
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from training.common import write_json  # type: ignore
    from training.v3 import DATA_DIR, REPORTS_DIR  # type: ignore
    from training.v3.data_sources import acquire, dataset_report_v3  # type: ignore
    from training.v3 import evaluation_v3, feature_analysis_v3, finalize, leaderboard, results_digest, risk_v3  # type: ignore
    from training.v3.quality import data_quality_audit, leakage_analysis  # type: ignore
    from training.v3.registry import dataset_manifest  # type: ignore


def _phase(summary: dict[str, Any], name: str, fn) -> None:
    print(f"\n=== {name} ===", flush=True)
    try:
        fn()
        summary[name] = "ok"
    except Exception:  # noqa: BLE001
        summary[name] = "failed"
        print(f"[run_v3] {name} FAILED:\n{traceback.format_exc()}", flush=True)


def main(args: argparse.Namespace) -> None:
    summary: dict[str, Any] = {}
    if args.acquire:
        _phase(summary, "phase1_acquire", lambda: acquire.acquire(acquire.build_parser().parse_args([])))
    _phase(summary, "phase1_dataset_report", lambda: dataset_report_v3.build(dataset_report_v3.build_parser().parse_args([])))
    _phase(summary, "phase2_data_quality", lambda: data_quality_audit.build(data_quality_audit.build_parser().parse_args([])))
    _phase(summary, "phase2_leakage", lambda: leakage_analysis.run(leakage_analysis.build_parser().parse_args([])))
    _phase(summary, "phase2_risk_v3_fix", lambda: risk_v3.run(risk_v3.build_parser().parse_args([])))
    _phase(summary, "phase3_features_shap", lambda: feature_analysis_v3.run(feature_analysis_v3.build_parser().parse_args([])))
    _phase(summary, "phase4_leaderboard", lambda: leaderboard.run(leaderboard.build_parser().parse_args([])))
    _phase(summary, "phase6_evaluation", lambda: evaluation_v3.run(evaluation_v3.build_parser().parse_args([])))

    # dataset version manifest (content hashes)
    paths = {p.stem: p for p in DATA_DIR.glob("*.csv")}
    paths["fake_reviews_v2"] = DATA_DIR.parent / "fake_reviews.csv"
    dataset_manifest({k: str(v) for k, v in paths.items()}, REPORTS_DIR / "dataset_versions.json")

    # finalize: confirm the chosen production model set loads + scores end-to-end
    _phase(summary, "finalize_production_set", lambda: finalize.run(finalize.build_parser().parse_args([])))
    # consolidated one-glance results
    _phase(summary, "results_digest", lambda: results_digest.build(results_digest.build_parser().parse_args([])))

    write_json(REPORTS_DIR / "v3_summary.json", summary)
    print(f"\n[run_v3] summary: {summary}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--acquire", action="store_true", help="download datasets first")
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
