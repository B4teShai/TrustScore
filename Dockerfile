FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TRUSTSCORE_MODEL_VERSION=best \
    AI_FEEDBACK_TIMEOUT_SECONDS=3

WORKDIR /app/apps/api

COPY apps/api/requirements-docker.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY apps/api/app ./app
COPY ml/artifacts/best /app/ml/artifacts/best
# Startup auto-migrations: the API creates/upgrades its schema on boot.
COPY db/migrations /app/db/migrations

EXPOSE 8080
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
