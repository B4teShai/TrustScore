"""FastAPI entrypoint for the AI TrustScore backend."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.product_analysis import router as product_analysis_router
from app.core.config import settings


app = FastAPI(
    title=settings.app_name,
    version=settings.model_version,
    description="Backend API for the AI TrustScore browser extension prototype.",
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
