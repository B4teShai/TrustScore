# Trainer / artifact-assembly container. Build context: repo root.
# On `docker compose up` it (re)assembles ml/artifacts/best/ from the
# version-by-version winners. Long training is run on demand, e.g.:
#   docker compose run --rm trainer python -m ml.training.train_all
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY ml/requirements.txt /app/ml/requirements.txt
RUN pip install --no-cache-dir -r /app/ml/requirements.txt

COPY ml /app/ml

CMD ["python", "-m", "ml.training.make_best_artifacts"]
