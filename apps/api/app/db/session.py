"""Lazy SQLAlchemy engine setup for optional TrustScore persistence."""

from __future__ import annotations

from functools import lru_cache
import logging
import time
from typing import Any

from app.core.config import settings


logger = logging.getLogger(__name__)

# After a connection failure, skip the database for a while instead of paying
# the connect timeout on every scan (persistence is best-effort either way).
_ENGINE_FAILURE_COOLDOWN_SECONDS = 300.0
_engine_failed_at: float | None = None


def report_engine_connection_failure() -> None:
    """Record a connection failure so scans stop retrying the database for a while."""
    global _engine_failed_at
    _engine_failed_at = time.monotonic()
    logger.warning(
        "database_connection_cooldown",
        extra={"cooldown_seconds": _ENGINE_FAILURE_COOLDOWN_SECONDS},
    )


def engine_in_cooldown() -> bool:
    if _engine_failed_at is None:
        return False
    if time.monotonic() - _engine_failed_at >= _ENGINE_FAILURE_COOLDOWN_SECONDS:
        return False
    return True


@lru_cache(maxsize=1)
def get_engine() -> Any | None:
    """Return a SQLAlchemy engine when DATABASE_URL is configured."""
    if not settings.database_url:
        return None

    try:
        from sqlalchemy import create_engine
    except Exception as exc:  # pragma: no cover - dependency issue only
        logger.warning(
            "database_dependency_unavailable",
            extra={"error_type": type(exc).__name__, "error": str(exc)},
            exc_info=True,
        )
        return None

    try:
        connect_args: dict[str, Any] = {}
        if settings.database_url.startswith(("postgres://", "postgresql")):
            # Persistence is best-effort: an unreachable database must cost a
            # few seconds per scan at most, not the OS TCP timeout (~75s).
            connect_args["connect_timeout"] = 3
        return create_engine(
            settings.database_url,
            pool_pre_ping=True,
            connect_args=connect_args,
        )
    except Exception as exc:
        logger.warning(
            "database_engine_unavailable",
            extra={"error_type": type(exc).__name__, "error": str(exc)},
            exc_info=True,
        )
        return None


def database_enabled() -> bool:
    """Return whether persistence is configured for this runtime."""
    return get_engine() is not None
