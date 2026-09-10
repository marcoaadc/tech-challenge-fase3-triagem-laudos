"""Metadados versionados do modelo: o contrato entre a etapa de treino e a API de inferencia."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

METADATA_FILENAME = "metadata.json"
SKLEARN_FILENAME = "model.joblib"
ONNX_FILENAME = "model.onnx"
ONNX_INT8_FILENAME = "model.int8.onnx"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class ModelMetadata:
    """Descreve um modelo treinado. Serializado em ``models/metadata.json``."""

    model_version: str
    model_type: str
    classes: list[str]
    trained_at: str
    dataset_rows: int
    dataset_sha256: str
    hyperparameters: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    artifacts: dict[str, dict] = field(default_factory=dict)
    seed: int = 42
    package_version: str = ""

    @staticmethod
    def now_iso() -> str:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    def register_artifact(self, name: str, path: Path) -> None:
        path = Path(path)
        self.artifacts[name] = {
            "filename": path.name,
            "sha256": sha256_of(path),
            "size_bytes": path.stat().st_size,
        }

    def save(self, models_dir: Path) -> Path:
        models_dir = Path(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        target = models_dir / METADATA_FILENAME
        target.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")
        return target

    @classmethod
    def load(cls, models_dir: Path) -> ModelMetadata:
        payload = json.loads((Path(models_dir) / METADATA_FILENAME).read_text(encoding="utf-8"))
        return cls(**payload)
