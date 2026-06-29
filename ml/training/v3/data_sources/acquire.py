"""Acquire freely-licensed HF datasets for v3 (curated subset, row-capped).

Each spec lists candidate HF ids (mirrors move around); the first that loads wins. Everything
is normalised to ``text`` (+ ``label`` for supervised sets) and saved to ``ml/data/v3/``.
Gated/manual sets (YelpChi, full Yelp) are documented, not downloaded.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from ml.training.common import clean_text, write_json
    from ml.training.v3 import DATA_DIR, SEED
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[3]))
    from training.common import clean_text, write_json  # type: ignore
    from training.v3 import DATA_DIR, SEED  # type: ignore


@dataclass
class DatasetSpec:
    key: str
    task: str
    license: str
    domain: str
    candidates: list[dict[str, Any]]  # each: {id, config?, split, text_cols, label_col?}
    cap: int | None = None
    notes: str = ""


# Candidate ids are tried in order; column mapping is per-candidate.
SPECS: list[DatasetSpec] = [
    DatasetSpec(
        key="ott_deceptive",
        task="fake_review",
        license="Research use (Ott et al. 2011/2013); CC-BY-NC commonly cited",
        domain="hotel reviews",
        cap=None,
        candidates=[
            {"id": "lukesjordan/deceptive-opinion-spam", "split": "train", "text_cols": ["text"], "label_col": "deceptive"},
            {"id": "kkfromus/deceptive-opinion-spam", "split": "train", "text_cols": ["text"], "label_col": "deceptive"},
            {"id": "deceptive-opinion", "split": "train", "text_cols": ["text"], "label_col": "deceptive"},
        ],
        notes="Gold human-labelled deceptive vs truthful — used as a cross-source OOD test for fake review.",
    ),
    DatasetSpec(
        key="amazon_polarity",
        task="sentiment",
        license="Apache-2.0 (HF card) / Amazon review data terms",
        domain="amazon product reviews",
        cap=100_000,
        candidates=[
            {"id": "fancyzhx/amazon_polarity", "split": "train", "text_cols": ["title", "content"], "label_col": "label"},
            {"id": "amazon_polarity", "split": "train", "text_cols": ["title", "content"], "label_col": "label"},
        ],
        notes="Binary polarity (0 neg / 1 pos).",
    ),
    DatasetSpec(
        key="yelp_polarity",
        task="sentiment",
        license="Yelp Dataset terms (research/personal)",
        domain="yelp business reviews",
        cap=100_000,
        candidates=[
            {"id": "fancyzhx/yelp_polarity", "split": "train", "text_cols": ["text"], "label_col": "label"},
            {"id": "yelp_polarity", "split": "train", "text_cols": ["text"], "label_col": "label"},
        ],
        notes="Binary polarity; cross-domain test vs Amazon.",
    ),
    DatasetSpec(
        key="imdb",
        task="sentiment",
        license="Research use (Maas et al. 2011)",
        domain="movie reviews",
        cap=50_000,
        candidates=[
            {"id": "stanfordnlp/imdb", "split": "train", "text_cols": ["text"], "label_col": "label"},
            {"id": "imdb", "split": "train", "text_cols": ["text"], "label_col": "label"},
        ],
        notes="Binary polarity; long-form, out-of-domain vs product reviews.",
    ),
    DatasetSpec(
        key="sst2",
        task="sentiment",
        license="Permissive (GLUE/SST-2)",
        domain="movie-review sentences",
        cap=None,
        candidates=[
            {"id": "stanfordnlp/sst2", "split": "train", "text_cols": ["sentence"], "label_col": "label"},
            {"id": "sst2", "split": "train", "text_cols": ["sentence"], "label_col": "label"},
            {"id": "glue", "config": "sst2", "split": "train", "text_cols": ["sentence"], "label_col": "label"},
        ],
        notes="Short-sentence sentiment; test labels are hidden (-1) so train/validation only.",
    ),
]

GATED = [
    {"key": "yelpchi", "reason": "YelpChi (Rayana & Akoglu) requires author request; not redistributable on HF."},
    {"key": "yelp_full", "reason": "Full Yelp Open Dataset requires accepting Yelp's dataset agreement; manual download."},
    {"key": "ieee_cis_fraud", "reason": "Kaggle competition data; license requires Kaggle login + competition rules acceptance."},
]


def _load_candidate(cand: dict[str, Any], cap: int | None) -> pd.DataFrame | None:
    from datasets import load_dataset

    kwargs: dict[str, Any] = {"split": cand["split"]}
    if cand.get("config"):
        args = (cand["id"], cand["config"])
    else:
        args = (cand["id"],)
    try:
        if cap:
            ds = load_dataset(*args, streaming=True, **kwargs)
            ds = ds.shuffle(seed=SEED, buffer_size=min(max(cap, 1000), 50_000))
            rows = list(ds.take(cap))
            frame = pd.DataFrame.from_records(rows)
        else:
            frame = load_dataset(*args, **kwargs).to_pandas()
    except Exception as exc:  # noqa: BLE001 - candidate may not exist
        print(f"    candidate {cand['id']} failed: {type(exc).__name__}: {str(exc)[:90]}")
        return None

    text_cols = [c for c in cand["text_cols"] if c in frame.columns]
    if not text_cols:
        print(f"    candidate {cand['id']} missing text cols {cand['text_cols']}; has {list(frame.columns)[:6]}")
        return None
    text = frame[text_cols[0]].fillna("").astype(str)
    for extra in text_cols[1:]:
        text = text + " " + frame[extra].fillna("").astype(str)
    out = pd.DataFrame({"text": text.map(clean_text)})
    if cand.get("label_col") and cand["label_col"] in frame.columns:
        out["label"] = frame[cand["label_col"]]
        out = out[out["label"].apply(lambda v: str(v) not in {"-1", "nan", "None"})]
    out = out[out["text"].str.len() > 0].reset_index(drop=True)
    return out


def acquire(args: argparse.Namespace) -> dict[str, Any]:
    manifest: dict[str, Any] = {"acquired": [], "unavailable": [], "gated": GATED}
    for spec in SPECS:
        print(f"[acquire] {spec.key} ({spec.task})")
        frame = None
        used_id = None
        for cand in spec.candidates:
            frame = _load_candidate(cand, spec.cap)
            if frame is not None and len(frame) > 0:
                used_id = cand["id"] + (f":{cand['config']}" if cand.get("config") else "")
                break
        if frame is None or len(frame) == 0:
            print(f"  -> UNAVAILABLE")
            manifest["unavailable"].append({"key": spec.key, "task": spec.task, "tried": [c["id"] for c in spec.candidates]})
            continue
        path = DATA_DIR / f"{spec.key}.csv"
        frame.to_csv(path, index=False)
        labels = frame["label"].value_counts().to_dict() if "label" in frame.columns else {}
        print(f"  -> {len(frame)} rows from {used_id}; labels={labels}")
        manifest["acquired"].append({
            "key": spec.key, "task": spec.task, "source_id": used_id, "rows": int(len(frame)),
            "license": spec.license, "domain": spec.domain, "label_distribution": {str(k): int(v) for k, v in labels.items()},
            "path": str(path), "notes": spec.notes,
        })
    write_json(DATA_DIR / "v3_dataset_manifest.json", manifest)
    print(f"[acquire] manifest -> {DATA_DIR / 'v3_dataset_manifest.json'}")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description=__doc__)


if __name__ == "__main__":
    acquire(build_parser().parse_args())
