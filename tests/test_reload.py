import shutil
from pathlib import Path

from fastapi.testclient import TestClient

from triage.api.main import create_app
from triage.config.settings import Settings
from triage.pipelines.export import run_export
from triage.pipelines.promote import Gates, run_promote
from triage.pipelines.train import run_train
from triage.training.metadata import ModelMetadata


def test_reload_serves_newly_promoted_model(trained_models_dir: Path, dataset_csv: Path, tmp_path: Path) -> None:
    serving_dir = tmp_path / "serving"
    shutil.copytree(trained_models_dir, serving_dir)
    settings = Settings(models_dir=serving_dir, _env_file=None)

    with TestClient(create_app(settings)) as client:
        first_version = client.get("/ready").json()["versao_modelo"]

        # Um "retreino" com outro seed produz uma versao diferente e a promove no mesmo diretorio.
        candidate = tmp_path / "candidate"
        run_train(dataset_csv, candidate, seed=11, candidates=["logreg"])
        run_export(candidate, dataset_csv, parity_samples=100)
        run_promote(candidate, serving_dir, Gates(min_f1_macro=0.80, min_urgent_recall=0.80))
        new_version = ModelMetadata.load(serving_dir).model_version
        assert new_version != first_version

        # Ate o reload, a API continua servindo a versao antiga.
        assert client.get("/ready").json()["versao_modelo"] == first_version

        body = client.post("/model/reload").json()
        assert body["status"] == "reloaded"
        assert body["versao_anterior"] == first_version
        assert body["versao_atual"] == new_version
        assert body["alterado"] is True
        assert client.get("/ready").json()["versao_modelo"] == new_version
        assert client.get("/model/info").json()["versao"] == new_version

        metrics = client.get("/metrics").text
        assert f'triage_model_info{{backend="onnx",type="tfidf+logreg",version="{new_version}"}} 1.0' in metrics
        assert f'version="{first_version}"' not in metrics
        assert 'triage_model_reloads_total{result="success"} 1.0' in metrics


def test_reload_failure_keeps_current_model(trained_models_dir: Path, tmp_path: Path) -> None:
    serving_dir = tmp_path / "serving"
    shutil.copytree(trained_models_dir, serving_dir)
    settings = Settings(models_dir=serving_dir, _env_file=None)

    with TestClient(create_app(settings)) as client:
        version = client.get("/ready").json()["versao_modelo"]
        (serving_dir / "model.onnx").unlink()  # simula artefato corrompido/ausente no disco

        response = client.post("/model/reload")
        assert response.status_code == 503
        assert "recarga falhou" in response.json()["detail"]
        # O modelo em memoria continua respondendo.
        assert client.get("/ready").json()["versao_modelo"] == version
        assert client.post("/predict", json={"texto": "exame normal"}).status_code == 200
        assert 'triage_model_reloads_total{result="failure"} 1.0' in client.get("/metrics").text


def test_reload_without_any_model_returns_503(tmp_path: Path) -> None:
    with TestClient(create_app(Settings(models_dir=tmp_path, _env_file=None))) as client:
        assert client.post("/model/reload").status_code == 503
