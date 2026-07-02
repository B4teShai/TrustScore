"""Hermetic environment for API tests.

app.core.config loads .env files (real API keys, a remote DATABASE_URL,
model-version overrides) with override=False, so any variable set here wins.
Pin the defaults tests assert against and blank out live-service credentials
before the app is imported. pytest's rootdir for these tests is apps/api
(pytest.ini), so this conftest runs before any test module import.
"""

import os


os.environ["TRUSTSCORE_MODEL_VERSION"] = "v3"
for _var in ("DATABASE_URL", "SERPER_API_KEY", "ANTHROPIC_API_KEY"):
    os.environ[_var] = ""
