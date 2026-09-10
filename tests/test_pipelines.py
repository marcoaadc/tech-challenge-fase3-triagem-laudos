import json
from pathlib import Path

import pytest

from triage.pipelines.ingest import run_ingest
from triage.pipelines.promote import Gates, PromotionGateError, check_gates, run_promote
from triage.training.metadata import ModelMetadata


def test_ingest_writes_csv_and_report(tmp_path: Path) -> None:
    report = run_ingest(tmp_path / "raw" / "laudos.csv", n_samples=2100, seed=1, reports_dir=tmp_path / "reports")
    assert (tmp_path / "raw" / "laudos.csv").exists()
    assert report["n_rows"] == 2100
    saved = json.loads((tmp_path / "reports" / "dataset_validation.json").read_text(encoding="utf-8"))
    assert saved["class_distribution"] == report["class_distribution"]


def test_promotion_registry_tracks_history(trained_models_dir: Path) -> None:
    registry = json.loads((trained_models_dir / "registry.json").read_text(encoding="utf-8"))
    meta = ModelMetadata.load(trained_models_dir)
    assert registry["current"] == meta.model_version
    assert registry["history"][-1]["onnx_sha256"] == meta.artifacts["onnx"]["sha256"]


def test_gate_rejects_weak_model(trained_models_dir: Path) -> None:
    meta = ModelMetadata.load(trained_models_dir)
    assert check_gates(meta, Gates(min_f1_macro=0.80, min_urgent_recall=0.80)) == []
    strict = Gates(min_f1_macro=0.999, min_urgent_recall=0.999)
    violations = check_gates(meta, strict)
    assert any("f1_macro" in v for v in violations)
    assert any("urgent_recall" in v for v in violations)


def test_gate_requires_onnx_artifact(trained_models_dir: Path) -> None:
    meta = ModelMetadata.load(trained_models_dir)
    meta.artifacts.pop("onnx")
    assert any("ONNX ausente" in v for v in check_gates(meta))


def test_run_promote_fails_closed(trained_models_dir: Path, tmp_path: Path) -> None:
    target = tmp_path / "prod"
    with pytest.raises(PromotionGateError):
        run_promote(trained_models_dir, target, Gates(min_f1_macro=0.999), reports_dir=tmp_path / "reports")
    assert not target.exists()
    decision = json.loads((tmp_path / "reports" / "promotion.json").read_text(encoding="utf-8"))
    assert decision["promoted"] is False and decision["violations"]
