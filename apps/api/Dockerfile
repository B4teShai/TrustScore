# API container. Build context: apps/api.
# The app is placed at /app/apps/api/app so that config.py resolves the repo
# root (parents[4]) to /app, where ml/artifacts is mounted by docker-compose.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app/apps/api

COPY requirements-docker.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
