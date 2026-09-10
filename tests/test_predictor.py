from pathlib import Path

import pytest

from triage.config.settings import Settings
from triage.serving.predictor import ModelNotFoundError, OnnxPredictor, SklearnPredictor, load_predictor
from triage.training.metadata import ModelMetadata

from .conftest import ATTENTION_TEXT, NORMAL_TEXT, URGENT_TEXT


@pytest.fixture(params=["onnx", "sklearn"])
def predictor(request, trained_models_dir: Path):
    settings = Settings(models_dir=trained_models_dir, model_backend=request.param, _env_file=None)
    return load_predictor(settings)


def test_factory_returns_requested_backend(trained_models_dir: Path) -> None:
    onnx = load_predictor(Settings(models_dir=trained_models_dir, model_backend="onnx", _env_file=None))
    sk = load_predictor(Settings(models_dir=trained_models_dir, model_backend="sklearn", _env_file=None))
    assert isinstance(onnx, OnnxPredictor) and onnx.backend == "onnx"
    assert isinstance(sk, SklearnPredictor) and sk.backend == "sklearn"


def test_predictions_are_well_formed(predictor) -> None:
    (pred,) = predictor.predict([URGENT_TEXT])
    assert pred.label == "urgente"
    assert set(pred.probabilities) == {"normal", "atencao", "urgente"}
    assert abs(sum(pred.probabilities.values()) - 1.0) < 1e-4
    assert pred.confidence == pytest.approx(max(pred.probabilities.values()))


def test_predictor_classifies_examples(predictor) -> None:
    labels = [p.label for p in predictor.predict([NORMAL_TEXT, ATTENTION_TEXT, URGENT_TEXT])]
    assert labels == ["normal", "atencao", "urgente"]


def test_empty_batch_returns_empty(predictor) -> None:
    assert predictor.predict([]) == []


def test_warmup_returns_milliseconds(predictor) -> None:
    assert predictor.warmup() >= 0.0


def test_backends_agree(trained_models_dir: Path, dataset_df) -> None:
    meta = ModelMetadata.load(trained_models_dir)
    onnx = OnnxPredictor(trained_models_dir, meta)
    sk = SklearnPredictor(trained_models_dir, meta)
    texts = dataset_df["text"].head(100).tolist()
    assert [p.label for p in onnx.predict(texts)] == [p.label for p in sk.predict(texts)]


def test_missing_artifacts_raise(tmp_path: Path) -> None:
    with pytest.raises(ModelNotFoundError):
        load_predictor(Settings(models_dir=tmp_path, _env_file=None))
    (tmp_path / "metadata.json").write_text("{}", encoding="utf-8")
    with pytest.raises((ModelNotFoundError, TypeError)):
        OnnxPredictor(tmp_path)
