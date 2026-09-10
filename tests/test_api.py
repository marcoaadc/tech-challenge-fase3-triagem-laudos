from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from triage.api.main import create_app
from triage.config.settings import Settings

from .conftest import NORMAL_TEXT, URGENT_TEXT


@pytest.fixture
def client(settings_onnx: Settings):
    with TestClient(create_app(settings_onnx)) as client:
        yield client


@pytest.fixture
def client_without_model(tmp_path: Path):
    settings = Settings(models_dir=tmp_path, _env_file=None)
    with TestClient(create_app(settings)) as client:
        yield client


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "X-Request-ID" in response.headers and "X-Process-Time-Ms" in response.headers


def test_ready_reports_loaded_model(client: TestClient) -> None:
    body = client.get("/ready").json()
    assert body["modelo_carregado"] is True
    assert body["backend"] == "onnx"


def test_predict_urgent(client: TestClient) -> None:
    response = client.post("/predict", json={"texto": URGENT_TEXT})
    assert response.status_code == 200
    body = response.json()
    assert body["classe"] == "urgente"
    assert body["confianca"] > 0.5
    assert set(body["probabilidades"]) == {"normal", "atencao", "urgente"}
    assert body["latencia_ms"] >= 0
    assert body["modelo"]["backend"] == "onnx"


def test_predict_normal(client: TestClient) -> None:
    assert client.post("/predict", json={"texto": NORMAL_TEXT}).json()["classe"] == "normal"


def test_predict_rejects_empty_text(client: TestClient) -> None:
    assert client.post("/predict", json={"texto": ""}).status_code == 422
    assert client.post("/predict", json={}).status_code == 422


def test_predict_rejects_oversized_text(settings_onnx: Settings) -> None:
    settings = settings_onnx.model_copy(update={"max_text_length": 100})
    with TestClient(create_app(settings)) as client:
        assert client.post("/predict", json={"texto": "x" * 101}).status_code == 413


def test_batch_predict(client: TestClient) -> None:
    response = client.post("/predict/batch", json={"laudos": [NORMAL_TEXT, URGENT_TEXT]})
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 2
    assert [r["classe"] for r in body["resultados"]] == ["normal", "urgente"]


def test_batch_limits(settings_onnx: Settings) -> None:
    settings = settings_onnx.model_copy(update={"max_batch_size": 2})
    with TestClient(create_app(settings)) as client:
        assert client.post("/predict/batch", json={"laudos": ["a", "b", "c"]}).status_code == 413
        assert client.post("/predict/batch", json={"laudos": ["a", "  "]}).status_code == 422
        assert client.post("/predict/batch", json={"laudos": []}).status_code == 422


def test_model_info(client: TestClient) -> None:
    body = client.get("/model/info").json()
    assert body["tipo"].startswith("tfidf+")
    assert body["classes"] == ["atencao", "normal", "urgente"]
    assert "test" in body["metricas"]


def test_metrics_endpoint_exposes_counters(client: TestClient) -> None:
    client.post("/predict", json={"texto": URGENT_TEXT})
    text = client.get("/metrics").text
    assert 'triage_http_requests_total{method="POST",path="/predict",status="200"}' in text
    assert 'triage_predictions_total{label="urgente"}' in text
    assert "triage_model_inference_duration_seconds_bucket" in text
    assert "triage_http_request_duration_seconds_bucket" in text
    assert "triage_model_info{" in text


def test_request_id_is_propagated(client: TestClient) -> None:
    response = client.get("/health", headers={"X-Request-ID": "abc-123"})
    assert response.headers["X-Request-ID"] == "abc-123"


def test_api_without_model_degrades_gracefully(client_without_model: TestClient) -> None:
    assert client_without_model.get("/health").status_code == 200
    assert client_without_model.get("/ready").status_code == 503
    assert client_without_model.post("/predict", json={"texto": "x"}).status_code == 503
    assert client_without_model.get("/model/info").status_code == 503


def test_openapi_documents_routes(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/health", "/ready", "/predict", "/predict/batch", "/model/info"} <= set(paths)
