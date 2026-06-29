"""Pytest import path setup for the mixed API/ML repository."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
API_ROOT = ROOT / "apps" / "api"

for path in (ROOT, API_ROOT):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

