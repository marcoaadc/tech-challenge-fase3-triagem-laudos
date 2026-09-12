# syntax=docker/dockerfile:1
#
# Imagem do servico de inferencia (API FastAPI + ONNX Runtime).
#
#   1. builder : instala o Poetry e resolve apenas as dependencias de runtime num virtualenv
#   2. runtime : imagem slim final, sem toolchain, usuario nao-root, healthcheck embutido
#
# Build:  docker build -t triage-api:latest .
# Run:    docker run --rm -p 8000:8000 triage-api:latest
#
# Os artefatos do modelo (models/) sao copiados para a imagem, tornando-a autossuficiente.
# Para servir um modelo re-treinado sem rebuild, monte um volume em /app/models.

# --------------------------------------------------------------------------
# Stage 1: builder
# --------------------------------------------------------------------------
FROM python:3.14-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_VERSION=2.1.3

RUN pip install "poetry==${POETRY_VERSION}"

WORKDIR /app

# Somente os manifests primeiro: a camada de dependencias so e invalidada quando eles mudam.
COPY pyproject.toml poetry.lock README.md ./
RUN poetry install --only main --no-root --no-ansi

# --------------------------------------------------------------------------
# Stage 2: runtime
# --------------------------------------------------------------------------
FROM python:3.14-slim AS runtime

ARG APP_VERSION=dev
LABEL org.opencontainers.image.title="triage-api" \
      org.opencontainers.image.description="Triagem automatica de laudos medicos (FastAPI + ONNX Runtime)" \
      org.opencontainers.image.version="${APP_VERSION}"

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TRIAGE_ENVIRONMENT=docker \
    TRIAGE_LOG_JSON=true \
    TRIAGE_MODEL_BACKEND=onnx \
    TRIAGE_MODELS_DIR=/app/models \
    TRIAGE_HOST=0.0.0.0 \
    TRIAGE_PORT=8000

RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --create-home appuser \
    && apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY src/ ./src/
COPY models/metadata.json models/model.onnx models/model.joblib ./models/

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:8000/ready || exit 1

CMD ["uvicorn", "triage.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
