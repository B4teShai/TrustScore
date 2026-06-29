"""Lightweight experiment tracking + reproducibility helpers for v3.

No external tracking server. Experiments append to ``ml/reports/v3/experiments.jsonl``;
datasets are content-hashed for version tracking. Deterministic given fixed seeds.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from ml.training.common import now_iso, write_json
    from ml.training.v3 import REPORTS_DIR
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from training.common import now_iso, write_json  # type: ignore
    from training.v3 import REPORTS_DIR  # type: ignore


EXPERIMENTS_LOG = REPORTS_DIR / "experiments.jsonl"


def content_hash(path: str | Path, *, max_bytes: int = 64 * 1024 * 1024) -> str:
    """SHA-256 of a file (capped read for very large files) — a dataset version id."""
    p = Path(path)
    if not p.exists():
        return "missing"
    h = hashlib.sha256()
    read = 0
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
            read += len(chunk)
            if read >= max_bytes:
                h.update(b"__truncated__")
                break
    return f"sha256:{h.hexdigest()[:16]}"


def log_experiment(
    *,
    phase: str,
    name: str,
    params: dict[str, Any],
    metrics: dict[str, Any],
    datasets: dict[str, str] | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """Append one experiment record to the JSONL registry and return it."""
    record = {
        "logged_at": now_iso(),
        "phase": phase,
        "name": name,
        "params": params,
        "metrics": metrics,
        "datasets": datasets or {},
        "notes": notes,
    }
    EXPERIMENTS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with EXPERIMENTS_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")
    return record


def write_model_card(
    *,
    model_id: str,
    task: str,
    intended_use: str,
    training_data: str,
    metrics: dict[str, Any],
    limitations: list[str],
    ethical_notes: list[str],
    out_dir: Path,
) -> Path:
    """Emit a Markdown model card."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{model_id}.md"
    lines = [
        f"# Model Card — {model_id}",
        "",
        f"- **Task:** {task}",
        f"- **Version:** v3 (0.3.0)",
        f"- **Generated:** {now_iso()}",
        "",
        "## Intended use",
        intended_use,
        "",
        "## Training data",
        training_data,
        "",
        "## Metrics",
        "```json",
        json.dumps(metrics, indent=2, sort_keys=True),
        "```",
        "",
        "## Limitations",
        *[f"- {x}" for x in limitations],
        "",
        "## Ethical considerations",
        *[f"- {x}" for x in ethical_notes],
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def dataset_manifest(paths: dict[str, str | Path], out_path: Path) -> dict[str, Any]:
    """Write a dataset version manifest (content hashes) and return it."""
    entries = {name: {"path": str(p), "hash": content_hash(p)} for name, p in paths.items()}
    manifest = {"created_at": now_iso(), "datasets": entries}
    write_json(out_path, manifest)
    return manifest
