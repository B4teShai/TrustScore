"""Orchestrate the v2 pipeline: dataset report -> baselines -> train -> compare.

Writes v2 artifacts to ml/artifacts/v2/ (v1 in ml/artifacts/ is never touched) and a
comparison_v1_v2.json + figure under ml/reports/v2/. Reads actual row counts from each
trainer and logs them (no silent truncation).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    from ml.training.common import write_json
    from ml.training.v2 import build_baselines, dataset_report
    from ml.training.v2 import train_fake_review_model_v2 as fake_v2
    from ml.training.v2 import train_risk_models_v2 as risk_v2
    from ml.training.v2 import train_sentiment_model_v2 as sentiment_v2
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from common import write_json  # type: ignore
    from v2 import build_baselines, dataset_report  # type: ignore
    from v2 import train_fake_review_model_v2 as fake_v2  # type: ignore
    from v2 import train_risk_models_v2 as risk_v2  # type: ignore
    from v2 import train_sentiment_model_v2 as sentiment_v2  # type: ignore


def _repo() -> Path:
    return Path(__file__).resolve().parents[2]


def _load(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def build_comparison(report_dir: Path) -> dict[str, Any]:
    art = _repo() / "artifacts"
    v1_fake = _load(art / "model_metadata.json").get("metrics", {})
    v1_sent = _load(art / "sentiment" / "sentiment_model_metadata.json").get("metrics", {})
    v1_risk = _load(art / "risk" / "risk_model_metadata.json").get("metrics", {})
    v2_fake = _load(art / "v2" / "model_metadata.json").get("metrics", {}).get("test", {})
    v2_sent = _load(art / "v2" / "sentiment" / "sentiment_model_metadata.json").get("metrics", {}).get("test", {})
    v2_risk = _load(art / "v2" / "risk" / "risk_model_metadata.json").get("metrics", {})

    def risk_v1(key: str) -> dict[str, Any]:
        m = v1_risk.get(key, {})
        return m if isinstance(m, dict) else {}

    def risk_v2_full(key: str) -> dict[str, Any]:
        m = v2_risk.get(key, {})
        return m.get("numeric_plus_text", {}) if isinstance(m, dict) else {}

    def risk_v2_text(key: str) -> dict[str, Any]:
        m = v2_risk.get(key, {})
        return m.get("text_only_ablation", {}) if isinstance(m, dict) else {}

    rows = [
        {"model": "fake_review", "metric": "accuracy",
         "v1": v1_fake.get("accuracy"), "v2": v2_fake.get("accuracy")},
        {"model": "fake_review", "metric": "roc_auc",
         "v1": v1_fake.get("roc_auc"), "v2": v2_fake.get("roc_auc")},
        {"model": "sentiment", "metric": "accuracy",
         "v1": v1_sent.get("accuracy"), "v2": v2_sent.get("accuracy")},
        {"model": "sentiment", "metric": "f1_macro",
         "v1": v1_sent.get("f1_macro"), "v2": v2_sent.get("f1_macro")},
        {"model": "seller_reliability", "metric": "accuracy",
         "v1": risk_v1("seller_reliability").get("accuracy"),
         "v2_text_only": risk_v2_text("seller_reliability").get("accuracy"),
         "v2": risk_v2_full("seller_reliability").get("accuracy")},
        {"model": "price_safety", "metric": "accuracy",
         "v1": risk_v1("price_safety").get("accuracy"),
         "v2_text_only": risk_v2_text("price_safety").get("accuracy"),
         "v2": risk_v2_full("price_safety").get("accuracy")},
        {"model": "return_policy_clarity", "metric": "accuracy",
         "v1": None,
         "v2_text_only": risk_v2_text("return_policy_clarity").get("accuracy"),
         "v2": risk_v2_full("return_policy_clarity").get("accuracy")},
    ]
    comparison = {"rows": rows, "baselines": _load(report_dir / "baselines.json")}
    write_json(report_dir / "comparison_v1_v2.json", comparison)

    # Accuracy comparison figure (v1 vs v2)
    acc = [r for r in rows if r["metric"] == "accuracy"]
    labels = [r["model"] for r in acc]
    v1_vals = [r.get("v1") or 0 for r in acc]
    v2_vals = [r.get("v2") or 0 for r in acc]
    import numpy as np

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - 0.2, v1_vals, 0.4, label="v1", color="#3d5a80")
    ax.bar(x + 0.2, v2_vals, 0.4, label="v2", color="#1aa6a6")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("v1 vs v2 accuracy by model")
    ax.legend()
    for i, (a, b) in enumerate(zip(v1_vals, v2_vals)):
        if a:
            ax.text(i - 0.2, a + 0.01, f"{a:.2f}", ha="center", fontsize=8)
        if b:
            ax.text(i + 0.2, b + 0.01, f"{b:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(report_dir / "figures" / "comparison_accuracy.png", dpi=130)
    plt.close(fig)
    return comparison


def main(args: argparse.Namespace) -> None:
    report_dir = _repo() / "reports" / "v2"
    (report_dir / "figures").mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}

    if not args.skip_report:
        print("\n=== [1/4] dataset report ===")
        dataset_report.build(dataset_report.build_parser().parse_args([]))
    if not args.skip_baselines:
        print("\n=== [2/4] baselines ===")
        build_baselines.build(build_baselines.build_parser().parse_args([]))

    print("\n=== [3/4] train v2 models ===")
    summary["fake_review"] = fake_v2.train(fake_v2.build_parser().parse_args([]))
    summary["sentiment"] = sentiment_v2.train(sentiment_v2.build_parser().parse_args([]))
    summary["risk"] = risk_v2.train(risk_v2.build_parser().parse_args([]))

    print("\n=== [4/4] comparison ===")
    summary["comparison"] = build_comparison(report_dir)
    write_json(report_dir / "training_summary_v2.json", summary)
    print(f"\nDone. Artifacts in {_repo() / 'artifacts' / 'v2'}; reports in {report_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-report", action="store_true")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--max", action="store_true", help="kept for CLI parity; trainers use full local CSVs")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
