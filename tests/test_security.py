from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from triage.api.main import create_app
from triage.api.security import SlidingWindowRateLimiter
from triage.config.settings import Settings

from .conftest import NORMAL_TEXT, URGENT_TEXT


def test_rate_limiter_sliding_window() -> None:
    limiter = SlidingWindowRateLimiter(limit=2, window_seconds=10.0)
    assert limiter.allow("c1", now=0.0) == (True, 0.0)
    assert limiter.allow("c1", now=1.0) == (True, 0.0)
    allowed, retry = limiter.allow("c1", now=2.0)
    assert allowed is False and 7.9 < retry <= 8.0
    assert limiter.allow("c2", now=2.0)[0] is True  # chaves independentes
    assert limiter.allow("c1", now=10.5)[0] is True  # janela deslizou
    limiter.reset()
    assert limiter.allow("c1", now=11.0)[0] is True


@pytest.fixture
def secured_client(trained_models_dir: Path):
    settings = Settings(models_dir=trained_models_dir, api_key="segredo-123", rate_limit_per_minute=3, _env_file=None)
    with TestClient(create_app(settings)) as client:
        yield client


def test_predict_requires_api_key(secured_client: TestClient) -> None:
    response = secured_client.post("/predict", json={"texto": URGENT_TEXT})
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "ApiKey"
    assert (
        secured_client.post("/predict", json={"texto": URGENT_TEXT}, headers={"X-API-Key": "errada"}).status_code == 401
    )
    ok = secured_client.post("/predict", json={"texto": URGENT_TEXT}, headers={"X-API-Key": "segredo-123"})
    assert ok.status_code == 200 and ok.json()["classe"] == "urgente"


def test_operational_routes_stay_open(secured_client: TestClient) -> None:
    assert secured_client.get("/health").status_code == 200
    assert secured_client.get("/ready").status_code == 200
    assert secured_client.get("/metrics").status_code == 200
    assert secured_client.get("/model/info").status_code == 200


def test_reload_requires_api_key(secured_client: TestClient) -> None:
    assert secured_client.post("/model/reload").status_code == 401
    body = secured_client.post("/model/reload", headers={"X-API-Key": "segredo-123"}).json()
    assert body["status"] == "reloaded" and body["alterado"] is False


def test_rate_limit_returns_429_with_retry_after(secured_client: TestClient) -> None:
    headers = {"X-API-Key": "segredo-123"}
    statuses = [
        secured_client.post("/predict", json={"texto": NORMAL_TEXT}, headers=headers).status_code for _ in range(4)
    ]
    assert statuses == [200, 200, 200, 429]
    blocked = secured_client.post("/predict/batch", json={"laudos": [NORMAL_TEXT]}, headers=headers)
    assert blocked.status_code == 429
    assert int(blocked.headers["Retry-After"]) >= 1
    metrics = secured_client.get("/metrics").text
    assert 'triage_rate_limited_total{path="/predict"}' in metrics


def test_rate_limit_is_per_client(secured_client: TestClient) -> None:
    headers = {"X-API-Key": "segredo-123"}
    for _ in range(3):
        assert secured_client.post("/predict", json={"texto": NORMAL_TEXT}, headers=headers).status_code == 200
    other = secured_client.post(
        "/predict", json={"texto": NORMAL_TEXT}, headers={**headers, "X-Forwarded-For": "10.0.0.9, 10.0.0.1"}
    )
    assert other.status_code == 200
