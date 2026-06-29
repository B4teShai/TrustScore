"""Phase 3 — SHAP + ablation for interpretability.

SHAP is run on the **interpretable** linguistic feature set for fake-review detection
(embedding dims are not human-interpretable). An ablation compares feature sets
(linguistic / TF-IDF / embeddings / embeddings+linguistic) so the marginal value of each is
explicit. Outputs feature_importance_v3.{json,md} + figures.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, roc_auc_score  # noqa: E402

try:
    from ml.training.common import clean_text, safe_train_test_split, write_json
    from ml.training.v2.features import build_fake_review_vectorizer
    from ml.training.v3 import FIGURES_DIR, REPORTS_DIR, SEED
    from ml.training.v3.features_v3 import FEATURE_GAPS, embed_texts, linguistic_features
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from training.common import clean_text, safe_train_test_split, write_json  # type: ignore
    from training.v2.features import build_fake_review_vectorizer  # type: ignore
    from training.v3 import FIGURES_DIR, REPORTS_DIR, SEED  # type: ignore
    from training.v3.features_v3 import FEATURE_GAPS, embed_texts, linguistic_features  # type: ignore


def _load(cap: int) -> pd.DataFrame:
    path = Path(__file__).resolve().parents[2] / "data" / "fake_reviews.csv"
    df = pd.read_csv(path)
    df["text"] = df["text"].map(clean_text)
    df["label"] = df["label"].astype(int)
    df = df[df["text"] != ""].reset_index(drop=True)
    if len(df) > cap:
        df = df.sample(n=cap, random_state=SEED).reset_index(drop=True)
    return df


def _shap_linguistic(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    import shap
    from xgboost import XGBClassifier

    ltr = linguistic_features(train["text"])
    lte = linguistic_features(test["text"])
    model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=SEED,
                          n_jobs=-1, eval_metric="logloss")
    model.fit(ltr, train["label"])
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(lte)
    mean_abs = np.abs(sv).mean(axis=0)
    importance = dict(sorted(zip(ltr.columns, [round(float(x), 4) for x in mean_abs]),
                             key=lambda kv: kv[1], reverse=True))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(list(importance.keys())[::-1], list(importance.values())[::-1], color="#1aa6a6")
    ax.set_title("SHAP mean|value| — linguistic features (fake review)")
    ax.set_xlabel("mean |SHAP|")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "shap_fake_linguistic.png", dpi=130)
    plt.close(fig)
    return importance


def _ablation(train: pd.DataFrame, test: pd.DataFrame) -> list[dict[str, Any]]:
    ytr, yte = train["label"], test["label"]
    results = []

    def _eval(name: str, Xtr, Xte) -> None:
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
        clf.fit(Xtr, ytr)
        prob = clf.predict_proba(Xte)[:, 1]
        results.append({"feature_set": name,
                        "accuracy": round(float(accuracy_score(yte, (prob >= 0.5).astype(int))), 4),
                        "roc_auc": round(float(roc_auc_score(yte, prob)), 4)})

    ling_tr = linguistic_features(train["text"]).to_numpy()
    ling_te = linguistic_features(test["text"]).to_numpy()
    _eval("linguistic_only", ling_tr, ling_te)

    vec = build_fake_review_vectorizer(max_features=20000, word_ngram=2, char_min=3, char_max=5)
    Xtr = vec.fit_transform(train["text"])
    Xte = vec.transform(test["text"])
    _eval("tfidf_only", Xtr, Xte)

    Etr = embed_texts(train["text"].tolist())
    Ete = embed_texts(test["text"].tolist())
    _eval("embeddings_only", Etr, Ete)
    _eval("embeddings_plus_linguistic", np.hstack([Etr, ling_tr]), np.hstack([Ete, ling_te]))
    return results


def run(args: argparse.Namespace) -> dict[str, Any]:
    df = _load(args.cap)
    train, test = safe_train_test_split(df, "label", 0.2, SEED)
    print(f"[features] SHAP + ablation on {len(train)}/{len(test)} fake-review rows")
    shap_importance = _shap_linguistic(train, test)
    ablation = _ablation(train, test)
    report = {"task": "fake_review", "shap_linguistic_importance": shap_importance, "ablation": ablation,
              "feature_gaps": FEATURE_GAPS}
    write_json(REPORTS_DIR / "feature_importance_v3.json", report)
    _write_md(report)
    print(f"[features] -> {REPORTS_DIR / 'feature_importance_v3.md'}")
    return report


def _write_md(report: dict[str, Any]) -> None:
    L = ["# v3 Feature Analysis — fake review", "",
         "## SHAP — linguistic features (interpretable)", "",
         "Mean |SHAP| from an XGBoost on linguistic features only (figure: `figures/shap_fake_linguistic.png`).", "",
         "| Feature | mean \\|SHAP\\| |", "|---|---|"]
    for k, v in report["shap_linguistic_importance"].items():
        L.append(f"| {k} | {v} |")
    L += ["", "## Ablation — feature-set value (LogReg)", "",
          "| Feature set | Accuracy | ROC-AUC |", "|---|---|---|"]
    for r in report["ablation"]:
        L.append(f"| {r['feature_set']} | {r['accuracy']} | {r['roc_auc']} |")
    L += ["", "## Documented feature gaps (data does not support)", ""]
    for k, v in report["feature_gaps"].items():
        L.append(f"- **{k}**: {v}")
    (REPORTS_DIR / "feature_importance_v3.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cap", type=int, default=12000)
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
