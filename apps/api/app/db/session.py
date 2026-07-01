"""Lazy SQLAlchemy engine setup for optional TrustScore persistence."""

from __future__ import annotations

from functools import lru_cache
import logging
from typing import Any

from app.core.config import settings


logger = logging.getLogger(__name__)


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
        return create_engine(settings.database_url, pool_pre_ping=True)
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
