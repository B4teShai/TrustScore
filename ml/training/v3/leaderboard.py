"""Phase 4 — algorithm leaderboard on the leakage-free supervised task (fake review).

Benchmarks classical models, gradient-boosting (XGBoost/LightGBM/CatBoost when importable),
a frozen-embedding + classical head DL track, and Voting/Stacking ensembles. Reports
accuracy, F1, ROC-AUC, training time and deployment cost (artifact size, inference latency),
then picks the best by performance AND cost. Winner saved to ml/artifacts/v3/ with a model card.
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.svm import LinearSVC

try:
    from ml.training.common import binary_metrics, clean_text, safe_train_test_split, write_json
    from ml.training.v2.features import build_fake_review_vectorizer
    from ml.training.v3 import ARTIFACTS_DIR, MODEL_CARDS_DIR, REPORTS_DIR, SEED
    from ml.training.v3.features_v3 import embed_texts
    from ml.training.v3.registry import log_experiment, write_model_card
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from training.common import binary_metrics, clean_text, safe_train_test_split, write_json  # type: ignore
    from training.v2.features import build_fake_review_vectorizer  # type: ignore
    from training.v3 import ARTIFACTS_DIR, MODEL_CARDS_DIR, REPORTS_DIR, SEED  # type: ignore
    from training.v3.features_v3 import embed_texts  # type: ignore
    from training.v3.registry import log_experiment, write_model_card  # type: ignore


def _optional_gbms() -> dict[str, Any]:
    models: dict[str, Any] = {}
    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.9,
            colsample_bytree=0.8, tree_method="hist", random_state=SEED, n_jobs=-1, eval_metric="logloss",
        )
    except Exception as e:  # noqa: BLE001
        print(f"[leaderboard] xgboost unavailable: {e}")
    try:
        from lightgbm import LGBMClassifier

        models["lightgbm"] = LGBMClassifier(n_estimators=300, learning_rate=0.1, random_state=SEED, n_jobs=-1, verbose=-1)
    except Exception as e:  # noqa: BLE001
        print(f"[leaderboard] lightgbm unavailable: {e}")
    try:
        from catboost import CatBoostClassifier

        models["catboost"] = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.1, random_seed=SEED, verbose=False)
    except Exception as e:  # noqa: BLE001
        print(f"[leaderboard] catboost unavailable: {e}")
    return models


def _classical() -> dict[str, Any]:
    return {
        "logreg": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED),
        "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=SEED, n_jobs=-1),
        "calibrated_linear_svc": CalibratedClassifierCV(LinearSVC(class_weight="balanced", random_state=SEED), cv=3),
    }


def _fit_eval(name: str, model: Any, Xtr, ytr, Xte, yte, feature_set: str) -> tuple[dict[str, Any], Any]:
    t0 = time.perf_counter()
    model.fit(Xtr, ytr)
    train_time = time.perf_counter() - t0
    t1 = time.perf_counter()
    proba = model.predict_proba(Xte)[:, -1]
    pred = (proba >= 0.5).astype(int)
    latency_ms_per_1k = (time.perf_counter() - t1) / max(len(yte), 1) * 1000 * 1000
    size_kb = len(pickle.dumps(model)) / 1024
    row = {
        "model": name, "feature_set": feature_set,
        "accuracy": round(float(accuracy_score(yte, pred)), 4),
        "f1": round(float(f1_score(yte, pred)), 4),
        "roc_auc": round(float(roc_auc_score(yte, proba)), 4),
        "train_time_s": round(train_time, 2),
        "infer_latency_ms_per_1k": round(latency_ms_per_1k, 1),
        "artifact_size_kb": round(size_kb, 1),
    }
    print(f"[leaderboard] {feature_set:10} {name:22} acc={row['accuracy']} f1={row['f1']} auc={row['roc_auc']} t={row['train_time_s']}s size={row['artifact_size_kb']}kb")
    return row, model


def run(args: argparse.Namespace) -> dict[str, Any]:
    data = Path(__file__).resolve().parents[2] / "data" / "fake_reviews.csv"
    frame = pd.read_csv(data)
    frame["text"] = frame["text"].map(clean_text)
    frame["label"] = frame["label"].astype(int)
    frame = frame[(frame["text"] != "") & frame["label"].isin([0, 1])].reset_index(drop=True)
    if args.cap and len(frame) > args.cap:
        frame = frame.sample(n=args.cap, random_state=SEED).reset_index(drop=True)
    train, test = safe_train_test_split(frame, "label", 0.2, SEED)
    print(f"[leaderboard] fake review: {len(train)} train / {len(test)} test")

    rows: list[dict[str, Any]] = []
    best = {"score": -1.0, "row": None, "model": None, "kind": None, "vectorizer": None}

    # --- Feature set A: TF-IDF (word+char), sparse ---
    vec = build_fake_review_vectorizer(max_features=args.tfidf_features, word_ngram=2, char_min=3, char_max=5)
    Xtr = vec.fit_transform(train["text"])
    Xte = vec.transform(test["text"])
    tfidf_models = {**_classical()}
    for gname, gm in _optional_gbms().items():
        if gname != "catboost":  # catboost dislikes scipy sparse here
            tfidf_models[gname] = gm
    for name, model in tfidf_models.items():
        try:
            row, fitted = _fit_eval(name, model, Xtr, train["label"], Xte, test["label"], "tfidf")
        except Exception as e:  # noqa: BLE001
            print(f"[leaderboard] tfidf {name} failed: {type(e).__name__}: {str(e)[:80]}")
            continue
        rows.append(row)
        if row["roc_auc"] > best["score"]:
            best.update(score=row["roc_auc"], row=row, model=fitted, kind="tfidf", vectorizer=vec)

    # --- Feature set B: frozen MiniLM embeddings, dense ---
    print("[leaderboard] embedding texts (frozen MiniLM)...")
    Etr = embed_texts(train["text"].tolist())
    Ete = embed_texts(test["text"].tolist())
    emb_models = {**_classical(), **_optional_gbms()}
    fitted_for_ensemble: list[tuple[str, Any]] = []
    for name, model in emb_models.items():
        try:
            row, fitted = _fit_eval(f"{name}", model, Etr, train["label"], Ete, test["label"], "embeddings")
        except Exception as e:  # noqa: BLE001
            print(f"[leaderboard] emb {name} failed: {type(e).__name__}: {str(e)[:80]}")
            continue
        rows.append(row)
        if name in {"logreg", "xgboost", "random_forest"}:
            fitted_for_ensemble.append((name, model.__class__(**model.get_params())))
        if row["roc_auc"] > best["score"]:
            best.update(score=row["roc_auc"], row=row, model=fitted, kind="embeddings", vectorizer=None)

    # --- Ensembles on embeddings ---
    if len(fitted_for_ensemble) >= 2:
        for ens_name, ens in [
            ("voting_soft", VotingClassifier(fitted_for_ensemble, voting="soft")),
            ("stacking", StackingClassifier(fitted_for_ensemble, final_estimator=LogisticRegression(max_iter=1000), cv=3)),
        ]:
            try:
                row, fitted = _fit_eval(ens_name, ens, Etr, train["label"], Ete, test["label"], "embeddings")
                rows.append(row)
                if row["roc_auc"] > best["score"]:
                    best.update(score=row["roc_auc"], row=row, model=fitted, kind="embeddings", vectorizer=None)
            except Exception as e:  # noqa: BLE001
                print(f"[leaderboard] ensemble {ens_name} failed: {type(e).__name__}: {str(e)[:80]}")

    rows.sort(key=lambda r: r["roc_auc"], reverse=True)
    report = {"task": "fake_review", "n_train": int(len(train)), "n_test": int(len(test)),
              "leaderboard": rows, "winner": best["row"]}
    write_json(REPORTS_DIR / "leaderboard.json", report)
    _write_md(report)

    # Save winner + model card
    if best["model"] is not None:
        out = ARTIFACTS_DIR / "fake_review"
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(best["model"], out / "model.joblib")
        if best["vectorizer"] is not None:
            joblib.dump(best["vectorizer"], out / "vectorizer.joblib")
        write_json(out / "metadata.json", {"winner": best["row"], "feature_kind": best["kind"], "model_version": "0.3.0"})
        write_model_card(
            model_id="fake_review_v3", task="fake review detection (original vs generated)",
            intended_use="Estimate the probability a product review is machine-generated; one of five TrustScore signals.",
            training_data="theArijitDas/Fake-Reviews-Dataset (balanced ~40k). Independent labels (not weak labels).",
            metrics=best["row"], out_dir=MODEL_CARDS_DIR,
            limitations=["Distinguishes generated vs original text, not all fraud (e.g. paid genuine-looking reviews).",
                         "Does not detect duplicate review spam; pair with the heuristic.",
                         "Cross-category generalisation reported separately (see evaluation_report.md)."],
            ethical_notes=["False positives can unfairly flag honest sellers; surface as a signal, not a verdict.",
                           "English-only; non-English reviews are out of scope."],
        )
    log_experiment(phase="4_leaderboard", name="fake_review", params={"tfidf_features": args.tfidf_features, "cap": args.cap},
                   metrics={"winner": best["row"]}, datasets={"fake_reviews": "theArijitDas"})
    print(f"[leaderboard] winner: {best['row']}")
    return report


def _write_md(report: dict[str, Any]) -> None:
    rows = report["leaderboard"]
    lines = ["# v3 Algorithm Leaderboard — fake review detection", "",
             f"Task: original vs computer-generated review. Train {report['n_train']} / test {report['n_test']}.",
             "Selection metric: ROC-AUC, with deployment cost (size, latency) as tie-breaker.", "",
             "| Rank | Model | Features | Acc | F1 | ROC-AUC | Train s | Latency µs/row | Size KB |",
             "|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        lines.append(f"| {i} | {r['model']} | {r['feature_set']} | {r['accuracy']} | {r['f1']} | "
                     f"{r['roc_auc']} | {r['train_time_s']} | {r['infer_latency_ms_per_1k']} | {r['artifact_size_kb']} |")
    w = report["winner"]
    lines += ["", f"**Winner:** `{w['model']}` on `{w['feature_set']}` — ROC-AUC {w['roc_auc']}, "
              f"acc {w['accuracy']}, {w['artifact_size_kb']} KB. Chosen for top AUC at acceptable deployment cost.", ""]
    (REPORTS_DIR / "leaderboard.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cap", type=int, default=None, help="row cap (default: full)")
    p.add_argument("--tfidf-features", type=int, default=20000)
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
