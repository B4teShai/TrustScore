"""Small environment loader for the first backend milestone."""

from dataclasses import dataclass
import os
from pathlib import Path


# Load apps/api/.env (where the user pastes ANTHROPIC_API_KEY) before any
# os.getenv read below. Real environment variables still win over the file.
def _load_env_file() -> None:
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.is_file():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path, override=False)
    except Exception:
        # python-dotenv missing: fall back to a tiny KEY=VALUE parser.
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_env_file()


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _get_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# Model version selection. v3 is the default because the v3 reports reject the
# leaked v2 risk classifiers and select a leakage-safe production model set.
_MODEL_VERSION_TAG = os.getenv("TRUSTSCORE_MODEL_VERSION", "v3").strip().lower()
_IS_BEST = _MODEL_VERSION_TAG in {"best", "prod", "production", "1.0.0"}
_IS_V2 = _MODEL_VERSION_TAG in {"v2", "0.2.0", "2"}
_IS_V3 = _MODEL_VERSION_TAG in {"v3", "0.3.0", "3"}
# The "best" set is curated from the v3 production semantics (rules + anomaly),
# so it shares every v3 conditional except the artifact base directory and the
# sentiment path (which lives inside best/ rather than pointing back at v2).
_IS_V3_LIKE = _IS_V3 or _IS_BEST
_REPO_ROOT = Path(__file__).resolve().parents[4]
_ARTIFACTS_BASE = (
    str(_REPO_ROOT / "ml" / "artifacts" / "best")
    if _IS_BEST
    else str(_REPO_ROOT / "ml" / "artifacts" / "v3")
    if _IS_V3
    else str(_REPO_ROOT / "ml" / "artifacts" / "v2")
    if _IS_V2
    else str(_REPO_ROOT / "ml" / "artifacts")
)
_DEFAULT_MODEL_VERSION = (
    "1.0.0" if _IS_BEST else "0.3.0" if _IS_V3 else "0.2.0" if _IS_V2 else "0.1.0"
)


def _model_path(name: str, relative: str) -> str:
    """Per-model artifact path: explicit env var wins, else derived from the version base."""
    return os.getenv(name, f"{_ARTIFACTS_BASE}/{relative}")


def _optional_model_path(name: str, relative: str | None) -> str | None:
    """Return an explicit model path, a version default, or None for disabled models."""
    if name in os.environ:
        return os.environ[name]
    if relative is None:
        return None
    return f"{_ARTIFACTS_BASE}/{relative}"


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "AI TrustScore API")
    app_env: str = os.getenv("APP_ENV", "local")
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    cors_allow_origins: str = os.getenv(
        "CORS_ALLOW_ORIGINS",
        "http://localhost:5173,chrome-extension://YOUR_EXTENSION_ID",
    )
    database_url: str | None = os.getenv("DATABASE_URL")
    supabase_url: str | None = os.getenv("SUPABASE_URL")
    browser_id_hash_salt: str = os.getenv(
        "BROWSER_ID_HASH_SALT",
        "local-dev-trustscore-salt",
    )
    enable_rendered_fetch: bool = _get_bool("ENABLE_RENDERED_FETCH", False)

    # Optional market reference enrichment. SERPER_API_KEY must stay server-side;
    # the extension never receives or sends this key.
    serper_api_key: str | None = os.getenv("SERPER_API_KEY")
    serper_api_url: str = os.getenv("SERPER_API_URL", "https://google.serper.dev/shopping")
    market_reference_enabled: bool = _get_bool("MARKET_REFERENCE_ENABLED", True)
    market_reference_timeout_seconds: float = _get_float(
        "MARKET_REFERENCE_TIMEOUT_SECONDS", 4.0
    )
    market_reference_cache_ttl_seconds: float = _get_float(
        "MARKET_REFERENCE_CACHE_TTL_SECONDS", 21600.0
    )
    market_reference_max_results: int = int(os.getenv("MARKET_REFERENCE_MAX_RESULTS", "12"))
    market_reference_min_results: int = int(os.getenv("MARKET_REFERENCE_MIN_RESULTS", "3"))
    exchange_rate_enabled: bool = _get_bool("EXCHANGE_RATE_ENABLED", True)
    exchange_rate_api_url: str = os.getenv(
        "EXCHANGE_RATE_API_URL",
        "https://api.frankfurter.dev/v2",
    )
    exchange_rate_timeout_seconds: float = _get_float("EXCHANGE_RATE_TIMEOUT_SECONDS", 3.0)
    exchange_rate_cache_ttl_seconds: float = _get_float(
        "EXCHANGE_RATE_CACHE_TTL_SECONDS",
        21600.0,
    )

    # AI feedback (Claude). When ANTHROPIC_API_KEY is set, the natural-language
    # shopping guidance is generated by Claude; otherwise the deterministic
    # rule-based recommendation is used. Generation is always best-effort.
    anthropic_api_key: str | None = os.getenv("ANTHROPIC_API_KEY")
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")
    ai_feedback_enabled: bool = _get_bool("AI_FEEDBACK_ENABLED", True)
    ai_feedback_max_tokens: int = int(os.getenv("AI_FEEDBACK_MAX_TOKENS", "320"))
    ai_feedback_timeout_seconds: float = _get_float("AI_FEEDBACK_TIMEOUT_SECONDS", 3.0)

    # Local persistence fallback: when no DATABASE_URL is configured, each scored
    # page (and the AI feedback used) is still saved to this JSONL file so demos
    # keep a durable record of what was scored.
    persist_local_scans: bool = _get_bool("PERSIST_LOCAL_SCANS", True)
    local_scan_store_path: str = os.getenv(
        "LOCAL_SCAN_STORE_PATH",
        str(_REPO_ROOT / "apps" / "api" / ".data" / "scans.jsonl"),
    )
    model_version: str = os.getenv("MODEL_VERSION", _DEFAULT_MODEL_VERSION)
    sentiment_model_name: str = os.getenv(
        "SENTIMENT_MODEL_NAME",
        "distilbert-base-uncased-finetuned-sst-2-english",
    )
    fake_review_model_path: str | None = _optional_model_path(
        "FAKE_REVIEW_MODEL_PATH",
        "fake_review/model.joblib" if _IS_V3_LIKE else "fake_review_rf.joblib",
    )
    fake_review_vectorizer_path: str | None = _optional_model_path(
        "FAKE_REVIEW_VECTORIZER_PATH",
        "fake_review/vectorizer.joblib" if _IS_V3_LIKE else "fake_review_vectorizer.joblib",
    )
    sentiment_artifact_path: str | None = os.getenv(
        "SENTIMENT_ARTIFACT_PATH",
        str(_REPO_ROOT / "ml" / "artifacts" / "v2" / "sentiment" / "sentiment_tfidf_logreg.joblib")
        if _IS_V3
        else f"{_ARTIFACTS_BASE}/sentiment/sentiment_tfidf_logreg.joblib",
    )
    seller_reliability_model_path: str | None = _optional_model_path(
        "SELLER_RELIABILITY_MODEL_PATH",
        None if _IS_V3_LIKE else "risk/seller_reliability_tfidf_rf.joblib",
    )
    price_safety_model_path: str | None = _optional_model_path(
        "PRICE_SAFETY_MODEL_PATH",
        None if _IS_V3_LIKE else "risk/price_safety_tfidf_rf.joblib",
    )
    policy_clarity_model_path: str | None = _optional_model_path(
        "POLICY_CLARITY_MODEL_PATH",
        None if _IS_V3_LIKE else "risk/policy_clarity_tfidf_rf.joblib",
    )
    price_anomaly_model_path: str | None = _optional_model_path(
        "PRICE_ANOMALY_MODEL_PATH",
        "risk/price_anomaly_iforest.joblib" if _IS_V3_LIKE else None,
    )
    price_anomaly_medians_path: str | None = _optional_model_path(
        "PRICE_ANOMALY_MEDIANS_PATH",
        "risk/price_anomaly_medians.json" if _IS_V3_LIKE else None,
    )
    trust_weight_review_authenticity: float = _get_float(
        "TRUST_WEIGHT_REVIEW_AUTHENTICITY", 0.30
    )
    trust_weight_seller_reliability: float = _get_float(
        "TRUST_WEIGHT_SELLER_RELIABILITY", 0.20
    )
    trust_weight_sentiment: float = _get_float("TRUST_WEIGHT_SENTIMENT", 0.20)
    trust_weight_policy: float = _get_float("TRUST_WEIGHT_POLICY", 0.15)
    trust_weight_price: float = _get_float("TRUST_WEIGHT_PRICE", 0.10)
    trust_weight_feedback: float = _get_float("TRUST_WEIGHT_FEEDBACK", 0.05)

    @property
    def model_version_tag(self) -> str:
        return _MODEL_VERSION_TAG

    @property
    def ai_feedback_active(self) -> bool:
        """AI feedback is on only when enabled and an API key is configured."""
        return bool(self.ai_feedback_enabled and self.anthropic_api_key)

    @property
    def use_v3_risk(self) -> bool:
        return _IS_V3_LIKE

    @property
    def cors_origins(self) -> list[str]:
        return [
            origin.strip()
            for origin in self.cors_allow_origins.split(",")
            if origin.strip()
        ]


settings = Settings()
