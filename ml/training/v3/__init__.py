"""TrustScore v3 — research-grade, honesty-first branch (model_version 0.3.0).

Fully isolated from v1/v2: writes only to ``ml/artifacts/v3/``, ``ml/reports/v3/`` and
``ml/data/v3/``. v3 may import ``ml/training/common.py`` and ``ml/training/v2/features.py``
read-only, but never writes to v1/v2 paths.

The defining goal is NOT higher accuracy. v2 showed the seller/price risk "wins" are inflated
by weak-label leakage (the labels are deterministic functions of the numeric features fed to
the model). v3 proves that leakage, measures real out-of-domain / cross-source generalization,
benchmarks algorithms honestly, and rejects any metric that is a leakage shortcut.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "v3"
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "v3"
REPORTS_DIR = REPO_ROOT / "reports" / "v3"
FIGURES_DIR = REPORTS_DIR / "figures"
MODEL_CARDS_DIR = REPORTS_DIR / "model_cards"
MODEL_VERSION = "0.3.0"
SEED = 42

for _d in (DATA_DIR, ARTIFACTS_DIR, REPORTS_DIR, FIGURES_DIR, MODEL_CARDS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
