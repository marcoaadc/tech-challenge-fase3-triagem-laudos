"""Camada de inferencia: backends intercambiaveis (Strategy) atras de uma interface unica.

``OnnxPredictor`` e o backend de producao (baixa latencia). ``SklearnPredictor`` carrega o
pipeline original e existe para comparacao de latencia e como fallback. A API escolhe o
backend por configuracao (``TRIAGE_MODEL_BACKEND``), sem alterar codigo.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from triage.config.settings import Settings
from triage.nlp.preprocessing import normalize_text
from triage.training.metadata import ONNX_FILENAME, SKLEARN_FILENAME, ModelMetadata


class ModelNotFoundError(FileNotFoundError):
    """Artefato do modelo ausente no diretorio configurado."""


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    probabilities: dict[str, float]


@runtime_checkable
class Predictor(Protocol):
    backend: str
    classes: list[str]
    metadata: ModelMetadata

    def predict(self, texts: Sequence[str]) -> list[Prediction]: ...

    def warmup(self) -> float: ...


def _to_predictions(classes: Sequence[str], proba: np.ndarray) -> list[Prediction]:
    out: list[Prediction] = []
    for row in np.asarray(proba, dtype=np.float64):
        idx = int(np.argmax(row))
        out.append(
            Prediction(
                label=str(classes[idx]),
                confidence=float(row[idx]),
                probabilities={str(c): float(p) for c, p in zip(classes, row, strict=True)},
            )
        )
    return out


class _BasePredictor:
    backend = "base"

    def __init__(self, metadata: ModelMetadata) -> None:
        self.metadata = metadata
        self.classes: list[str] = list(metadata.classes)

    def _run(self, normalized: list[str]) -> np.ndarray:  # pragma: no cover - abstrato
        raise NotImplementedError

    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        if not texts:
            return []
        normalized = [normalize_text(t) for t in texts]
        return _to_predictions(self.classes, self._run(normalized))

    def warmup(self) -> float:
        """Executa uma inferencia descartavel para aquecer caches; retorna o tempo em ms."""
        start = time.perf_counter()
        self.predict(["laudo de aquecimento: exame sem alteracoes."])
        return (time.perf_counter() - start) * 1000.0


class SklearnPredictor(_BasePredictor):
    backend = "sklearn"

    def __init__(self, models_dir: Path, metadata: ModelMetadata | None = None) -> None:
        import joblib

        models_dir = Path(models_dir)
        path = models_dir / SKLEARN_FILENAME
        if not path.exists():
            raise ModelNotFoundError(f"modelo scikit-learn nao encontrado: {path}")
        super().__init__(metadata or ModelMetadata.load(models_dir))
        self.pipeline = joblib.load(path)
        # Garante alinhamento entre a ordem das classes do modelo e dos metadados.
        self.classes = [str(c) for c in self.pipeline.classes_]

    def _run(self, normalized: list[str]) -> np.ndarray:
        return self.pipeline.predict_proba(normalized)


class OnnxPredictor(_BasePredictor):
    backend = "onnx"

    def __init__(self, models_dir: Path, metadata: ModelMetadata | None = None, intra_op_threads: int = 1) -> None:
        import onnxruntime as ort

        models_dir = Path(models_dir)
        path = models_dir / ONNX_FILENAME
        if not path.exists():
            raise ModelNotFoundError(f"modelo ONNX nao encontrado: {path}")
        super().__init__(metadata or ModelMetadata.load(models_dir))

        options = ort.SessionOptions()
        # Otimizacoes de grafo completas (fusao de operadores, eliminacao de nos redundantes).
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Para requisicoes unitarias, 1 thread evita overhead de sincronizacao entre threads.
        options.intra_op_num_threads = intra_op_threads
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        self._input_name = self.session.get_inputs()[0].name
        self._proba_output = self.session.get_outputs()[-1].name

    def _run(self, normalized: list[str]) -> np.ndarray:
        batch = np.array(normalized, dtype=object).reshape(-1, 1)
        (proba,) = self.session.run([self._proba_output], {self._input_name: batch})
        return proba


def load_predictor(settings: Settings) -> Predictor:
    """Factory: instancia o backend configurado a partir do diretorio de modelos."""
    models_dir = Path(settings.models_dir)
    metadata_path = models_dir / "metadata.json"
    if not metadata_path.exists():
        raise ModelNotFoundError(f"metadata.json nao encontrado em {models_dir}; execute o pipeline de treino")
    metadata = ModelMetadata.load(models_dir)
    if settings.model_backend == "sklearn":
        return SklearnPredictor(models_dir, metadata)
    if settings.model_backend == "onnx":
        return OnnxPredictor(models_dir, metadata, intra_op_threads=settings.onnx_intra_op_threads)
    raise ValueError(f"backend desconhecido: {settings.model_backend}")
