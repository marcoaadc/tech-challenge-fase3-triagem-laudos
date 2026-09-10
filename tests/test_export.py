import json
from pathlib import Path

import joblib

from triage.nlp.preprocessing import normalize_text
from triage.training.export import check_parity, export_onnx
from triage.training.metadata import ONNX_FILENAME, SKLEARN_FILENAME, ModelMetadata, sha256_of


def test_onnx_export_matches_sklearn(trained_models_dir: Path, dataset_df, tmp_path) -> None:
    pipe = joblib.load(trained_models_dir / SKLEARN_FILENAME)
    onnx_path = export_onnx(pipe, tmp_path / "m.onnx")
    assert onnx_path.exists() and onnx_path.stat().st_size > 10_000

    texts = [normalize_text(t) for t in dataset_df["text"].head(200)]
    report = check_parity(pipe, onnx_path, texts)
    assert report.n_samples == 200
    assert report.label_agreement >= 0.99
    assert report.mean_abs_prob_diff < 0.005


def test_metadata_records_artifacts_and_parity(trained_models_dir: Path) -> None:
    meta = ModelMetadata.load(trained_models_dir)
    assert set(meta.artifacts) == {"sklearn", "onnx"}
    assert meta.artifacts["onnx"]["sha256"] == sha256_of(trained_models_dir / ONNX_FILENAME)
    assert meta.metrics["onnx_parity"]["label_agreement"] >= 0.99
    assert meta.classes == ["atencao", "normal", "urgente"]
    raw = json.loads((trained_models_dir / "metadata.json").read_text(encoding="utf-8"))
    assert raw["model_version"] == meta.model_version
