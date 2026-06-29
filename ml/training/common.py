"""Shared helpers for TrustScore training scripts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import html
import json
from pathlib import Path
import re
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


HTML_TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")
REQUIRED_FAKE_REVIEW_COLUMNS = ("text", "label")


@dataclass(frozen=True)
class DatasetInfo:
    """Dataset source details stored in model metadata."""

    name: str
    source: str
    url: str | None
    row_count: int


def repo_root() -> Path:
    """Return the repository root from the ml/training package."""
    return Path(__file__).resolve().parents[2]


def default_artifacts_dir() -> Path:
    return repo_root() / "ml" / "artifacts"


def clean_text(value: object) -> str:
    """Normalize review/product text for repeatable ML features."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    without_html = HTML_TAG_RE.sub(" ", html.unescape(str(value)))
    return WHITESPACE_RE.sub(" ", without_html).strip().lower()


def parse_float(value: object, default: float = 0.0) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    try:
        text = str(value).strip().replace("$", "").replace(",", "")
        if text.lower() in {"", "none", "nan", "null"}:
            return default
        return float(text)
    except (TypeError, ValueError):
        return default


def load_local_csv(path: str | Path) -> pd.DataFrame:
    """Load a local CSV with a clear error if it is missing."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Local CSV not found: {csv_path}")
    return pd.read_csv(csv_path)


def load_hf_data_files(
    builder_name: str,
    data_files: list[str],
    *,
    split: str,
    sample_size: int | None,
    seed: int,
) -> pd.DataFrame:
    """Load explicit Hugging Face data files without invoking dataset scripts."""
    from datasets import load_dataset

    files_by_split = {split: data_files}
    if sample_size is not None and sample_size > 0:
        dataset = load_dataset(
            builder_name,
            data_files=files_by_split,
            split=split,
            streaming=True,
        )
        buffer_size = min(max(sample_size, 1_000), 50_000)
        dataset = dataset.shuffle(seed=seed, buffer_size=buffer_size)
        return pd.DataFrame.from_records(list(dataset.take(sample_size)))

    dataset = load_dataset(builder_name, data_files=files_by_split, split=split)
    return dataset.to_pandas()


def require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")


def sample_frame(frame: pd.DataFrame, sample_size: int | None, seed: int) -> pd.DataFrame:
    if sample_size is None or sample_size <= 0 or sample_size >= len(frame):
        return frame.reset_index(drop=True)
    return frame.sample(n=sample_size, random_state=seed).reset_index(drop=True)


def safe_train_test_split(
    frame: pd.DataFrame,
    label_column: str,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split with stratification when the label distribution allows it."""
    labels = frame[label_column]
    value_counts = labels.value_counts()
    stratify = labels if len(value_counts) > 1 and value_counts.min() >= 2 else None
    train, test = train_test_split(
        frame,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def binary_metrics(y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray | None) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if y_prob is not None and len(set(y_true)) > 1:
        metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_prob)), 4)
    return metrics


def multiclass_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, Any]:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_macro": round(
            float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            4,
        ),
        "recall_macro": round(
            float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            4,
        ),
        "f1_macro": round(
            float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            4,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def dump_joblib(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path)


def now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def metadata_payload(
    *,
    model_name: str,
    model_version: str,
    dataset: DatasetInfo,
    metrics: dict[str, Any],
    params: dict[str, Any],
    artifacts: dict[str, str],
    limitations: list[str],
) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "model_version": model_version,
        "trained_at": now_iso(),
        "dataset": asdict(dataset),
        "metrics": metrics,
        "params": params,
        "artifacts": artifacts,
        "limitations": limitations,
    }
