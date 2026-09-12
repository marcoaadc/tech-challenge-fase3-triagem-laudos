#!/bin/sh
# Entrypoint do servico de inferencia.
#
# - TRIAGE_WORKERS=1 (padrao): um processo uvicorn, metricas Prometheus em memoria.
# - TRIAGE_WORKERS>1: varios processos; o prometheus_client precisa do modo multiprocesso
#   (PROMETHEUS_MULTIPROC_DIR) para que /metrics agregue todos os workers.
set -eu

WORKERS="${TRIAGE_WORKERS:-1}"
HOST="${TRIAGE_HOST:-0.0.0.0}"
PORT="${TRIAGE_PORT:-8000}"
LOG_LEVEL="$(printf '%s' "${TRIAGE_LOG_LEVEL:-info}" | tr '[:upper:]' '[:lower:]')"

if [ "$WORKERS" -gt 1 ]; then
    export PROMETHEUS_MULTIPROC_DIR="${PROMETHEUS_MULTIPROC_DIR:-/tmp/triage-metrics}"
    rm -rf "$PROMETHEUS_MULTIPROC_DIR"
    mkdir -p "$PROMETHEUS_MULTIPROC_DIR"
    echo "modo multiprocesso: ${WORKERS} workers, metricas em ${PROMETHEUS_MULTIPROC_DIR}"
fi

exec uvicorn triage.api.main:app \
    --host "$HOST" \
    --port "$PORT" \
    --workers "$WORKERS" \
    --log-level "$LOG_LEVEL" \
    --loop uvloop \
    --http httptools \
    "$@"
