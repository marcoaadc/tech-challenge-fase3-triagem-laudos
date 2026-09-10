"""Instrumentacao Prometheus da API.

Metricas expostas em ``/metrics`` (formato de exposicao do Prometheus):

- ``triage_http_requests_total{method,path,status}``: contagem de chamadas (metodo RED: Rate/Errors).
- ``triage_http_request_duration_seconds{method,path}``: histograma de latencia ponta a ponta (Duration).
- ``triage_http_requests_in_progress``: requisicoes em andamento.
- ``triage_model_inference_duration_seconds{backend}``: latencia so do modelo (sem overhead HTTP).
- ``triage_predictions_total{label}``: predicoes por classe (permite ver drift de distribuicao).
- ``triage_prediction_confidence``: histograma da confianca das predicoes.
- ``triage_model_info{version,type,backend}``: gauge=1 com as labels do modelo carregado.
- ``triage_exceptions_total{type}``: excecoes nao tratadas.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

REGISTRY = CollectorRegistry(auto_describe=True)

# Buckets pensados para uma API de baixa latencia (1 ms ... 5 s).
LATENCY_BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
INFERENCE_BUCKETS = (0.00005, 0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.5)

HTTP_REQUESTS_TOTAL = Counter(
    "triage_http_requests_total",
    "Total de requisicoes HTTP recebidas.",
    ["method", "path", "status"],
    registry=REGISTRY,
)
HTTP_REQUEST_DURATION = Histogram(
    "triage_http_request_duration_seconds",
    "Latencia ponta a ponta das requisicoes HTTP.",
    ["method", "path"],
    buckets=LATENCY_BUCKETS,
    registry=REGISTRY,
)
HTTP_IN_PROGRESS = Gauge(
    "triage_http_requests_in_progress",
    "Requisicoes HTTP em andamento.",
    registry=REGISTRY,
)
MODEL_INFERENCE_DURATION = Histogram(
    "triage_model_inference_duration_seconds",
    "Latencia da inferencia do modelo (sem overhead HTTP).",
    ["backend"],
    buckets=INFERENCE_BUCKETS,
    registry=REGISTRY,
)
PREDICTIONS_TOTAL = Counter(
    "triage_predictions_total",
    "Predicoes realizadas por classe de urgencia.",
    ["label"],
    registry=REGISTRY,
)
PREDICTION_CONFIDENCE = Histogram(
    "triage_prediction_confidence",
    "Distribuicao da confianca (probabilidade maxima) das predicoes.",
    buckets=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
    registry=REGISTRY,
)
MODEL_INFO = Gauge(
    "triage_model_info",
    "Informacoes do modelo carregado (valor sempre 1).",
    ["version", "type", "backend"],
    registry=REGISTRY,
)
EXCEPTIONS_TOTAL = Counter(
    "triage_exceptions_total",
    "Excecoes nao tratadas por tipo.",
    ["type"],
    registry=REGISTRY,
)


def observe_prediction(backend: str, label: str, confidence: float, inference_seconds: float) -> None:
    MODEL_INFERENCE_DURATION.labels(backend=backend).observe(inference_seconds)
    PREDICTIONS_TOTAL.labels(label=label).inc()
    PREDICTION_CONFIDENCE.observe(confidence)
