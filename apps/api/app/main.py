"""FastAPI entrypoint for the AI TrustScore backend."""

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.product_analysis import router as product_analysis_router
from app.core.config import settings
from app.db.migrations import run_startup_migrations


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Create/upgrade the persistence schema on boot when a database is
    # configured. Best-effort: the API must start (and score) even when the
    # database is missing or unreachable.
    if settings.auto_migrate:
        try:
            run_startup_migrations()
        except Exception:  # pragma: no cover - runner already logs failures
            logger.warning("startup_migrations_unexpected_error", exc_info=True)
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.model_version,
    description="Backend API for the AI TrustScore browser extension prototype.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(product_analysis_router)


@app.get("/health", tags=["system"])
def health_check() -> dict[str, str]:
    """Return a small response used by local setup and uptime checks."""
    return {"status": "ok", "service": "ai-trustscore-api"}
