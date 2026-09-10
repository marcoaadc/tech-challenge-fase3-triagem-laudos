from pathlib import Path

import pytest

from triage.config.settings import Settings


def test_defaults() -> None:
    s = Settings(_env_file=None)
    assert s.model_backend == "onnx"
    assert s.raw_dataset_path == Path("data") / "raw" / "laudos.csv"
    assert s.port == 8000


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRIAGE_MODEL_BACKEND", "sklearn")
    monkeypatch.setenv("TRIAGE_PORT", "9000")
    monkeypatch.setenv("TRIAGE_MODELS_DIR", "/opt/models")
    s = Settings(_env_file=None)
    assert s.model_backend == "sklearn"
    assert s.port == 9000
    assert s.models_dir == Path("/opt/models")


def test_invalid_backend_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRIAGE_MODEL_BACKEND", "tensorrt")
    with pytest.raises(ValueError):
        Settings(_env_file=None)
