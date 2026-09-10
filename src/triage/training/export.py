"""Exportacao do pipeline scikit-learn para ONNX e verificacao de paridade.

Tecnica de otimizacao de latencia adotada: **conversao para ONNX Runtime**. O grafo
exportado executa tokenizacao, TF-IDF e o classificador linear em C++ numa unica chamada,
eliminando o overhead Python do ``TfidfVectorizer.transform`` (regex + matriz esparsa) que
domina a latencia de inferencia unitaria no scikit-learn.

Quantizacao dinamica INT8 foi avaliada e descartada: o classificador e exportado como o
operador ``LinearClassifier`` (dominio ai.onnx.ml), que o quantizador do ONNX Runtime nao
cobre, e a variante com ``MatMul`` generico ficou mais lenta que a original. Para modelos
lineares pequenos o ganho vem do runtime, nao da precisao numerica dos pesos.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.pipeline import Pipeline

ONNX_INPUT_NAME = "text"
ONNX_OPSET = 17


def export_onnx(pipe: Pipeline, path: Path) -> Path:
    """Converte o pipeline (TF-IDF + classificador) para um arquivo ``.onnx``."""
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import StringTensorType

    clf = pipe.named_steps["clf"]
    onnx_model = convert_sklearn(
        pipe,
        name="triage-classifier",
        initial_types=[(ONNX_INPUT_NAME, StringTensorType([None, 1]))],
        # zipmap=False: probabilidades como tensor (N, n_classes) em vez de lista de dicts.
        options={id(clf): {"zipmap": False}},
        target_opset=ONNX_OPSET,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(onnx_model.SerializeToString())
    return path


@dataclass
class ParityReport:
    n_samples: int
    label_agreement: float
    max_abs_prob_diff: float
    mean_abs_prob_diff: float

    def as_dict(self) -> dict:
        return asdict(self)


def check_parity(pipe: Pipeline, onnx_path: Path, normalized_texts: Sequence[str]) -> ParityReport:
    """Compara predicoes do scikit-learn e do ONNX Runtime sobre os mesmos textos (ja normalizados)."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inputs = np.array(list(normalized_texts), dtype=object).reshape(-1, 1)
    onnx_labels, onnx_proba = session.run(None, {ONNX_INPUT_NAME: inputs})
    sk_proba = pipe.predict_proba(list(normalized_texts))
    sk_labels = pipe.predict(list(normalized_texts))

    diff = np.abs(np.asarray(onnx_proba, dtype=np.float64) - sk_proba)
    return ParityReport(
        n_samples=len(inputs),
        label_agreement=float(np.mean(np.asarray(onnx_labels).ravel() == sk_labels)),
        max_abs_prob_diff=float(diff.max()),
        mean_abs_prob_diff=float(diff.mean()),
    )
