# TechPulse Assignment 10 - prediction-service image.
# Lightweight, self-contained: the deployment artifact (joblib model) is
# generated deterministically INSIDE the image during build, so the
# container never depends on a model sitting on the developer's machine.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MLFLOW_ALLOW_FILE_STORE=true

WORKDIR /app

# Dependencies first for layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code only (tests/dev files excluded via .dockerignore).
COPY src/ ./src/
COPY scripts/ ./scripts/

# Deterministic smoke-sized training -> creates artifacts/ inside the image.
RUN python -m src.train --smoke

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD ["python", "scripts/healthcheck.py", "http://localhost:8000"]

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
