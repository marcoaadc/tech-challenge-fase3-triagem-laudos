"""Construcao, treino e avaliacao do classificador de urgencia (TF-IDF + modelo linear).

Decisao de modelagem: um modelo *leve* (bag-of-ngrams + classificador linear) atende ao
requisito de latencia (< 10 ms por laudo em CPU) e e integralmente conversivel para ONNX.
Modelos alternativos (Naive Bayes, Random Forest) sao treinados como candidatos para que a
escolha seja baseada em metrica de validacao, nao em preferencia previa.

A normalizacao de texto (``triage.nlp.normalize_text``) e aplicada FORA do pipeline do
scikit-learn, porque transformadores customizados nao sao conversiveis para ONNX. A API
aplica exatamente a mesma funcao antes da inferencia.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline

from triage.data.generator import LABELS
from triage.nlp.preprocessing import normalize_text

# Parametros do TF-IDF. ``token_pattern`` explicito (sem ``\b``) garante que o tokenizador do
# ONNX Runtime produza exatamente os mesmos n-gramas que o scikit-learn.
TFIDF_PARAMS: dict = {
    "ngram_range": (1, 2),
    "min_df": 2,
    "sublinear_tf": True,
    "lowercase": False,  # texto ja normalizado por normalize_text
    "strip_accents": None,
    "token_pattern": r"[a-z0-9][a-z0-9]+",
}

MODEL_CANDIDATES: dict[str, Callable[[int], ClassifierMixin]] = {
    "logreg": lambda seed: LogisticRegression(C=5.0, max_iter=2000, class_weight="balanced", random_state=seed),
    "complement_nb": lambda seed: ComplementNB(alpha=0.3),
    "random_forest": lambda seed: RandomForestClassifier(
        n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=seed
    ),
}

DEFAULT_MODEL = "logreg"


def build_pipeline(model_type: str = DEFAULT_MODEL, seed: int = 42) -> Pipeline:
    if model_type not in MODEL_CANDIDATES:
        raise ValueError(f"model_type desconhecido: {model_type!r}; opcoes: {sorted(MODEL_CANDIDATES)}")
    return Pipeline([("tfidf", TfidfVectorizer(**TFIDF_PARAMS)), ("clf", MODEL_CANDIDATES[model_type](seed))])


def prepare_texts(texts: Sequence[str]) -> list[str]:
    return [normalize_text(t) for t in texts]


def train_pipeline(pipe: Pipeline, texts: Sequence[str], labels: Sequence[str]) -> Pipeline:
    """Treina o pipeline sobre textos crus (a normalizacao e aplicada aqui)."""
    pipe.fit(prepare_texts(texts), list(labels))
    return pipe


@dataclass
class EvaluationResult:
    accuracy: float
    f1_macro: float
    f1_weighted: float
    per_class: dict[str, dict[str, float]]
    confusion_matrix: list[list[int]]
    labels: list[str] = field(default_factory=lambda: list(LABELS))
    report_text: str = ""

    @property
    def urgent_recall(self) -> float:
        """Recall da classe ``urgente``: a metrica clinicamente critica (falso negativo = risco)."""
        return self.per_class["urgente"]["recall"]

    def flat_metrics(self, prefix: str = "") -> dict[str, float]:
        out = {
            f"{prefix}accuracy": self.accuracy,
            f"{prefix}f1_macro": self.f1_macro,
            f"{prefix}f1_weighted": self.f1_weighted,
            f"{prefix}urgent_recall": self.urgent_recall,
        }
        for lb, m in self.per_class.items():
            for k, v in m.items():
                out[f"{prefix}{lb}_{k}"] = v
        return out

    def as_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "f1_macro": self.f1_macro,
            "f1_weighted": self.f1_weighted,
            "urgent_recall": self.urgent_recall,
            "per_class": self.per_class,
            "confusion_matrix": self.confusion_matrix,
            "labels": self.labels,
        }


def evaluate_pipeline(pipe: Pipeline, texts: Sequence[str], labels: Sequence[str]) -> EvaluationResult:
    y_true = list(labels)
    y_pred = pipe.predict(prepare_texts(texts))
    label_list = list(LABELS)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=label_list, zero_division=0)
    per_class = {
        lb: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
        for i, lb in enumerate(label_list)
    }
    return EvaluationResult(
        accuracy=float(accuracy_score(y_true, y_pred)),
        f1_macro=float(f1_score(y_true, y_pred, average="macro", labels=label_list)),
        f1_weighted=float(f1_score(y_true, y_pred, average="weighted", labels=label_list)),
        per_class=per_class,
        confusion_matrix=confusion_matrix(y_true, y_pred, labels=label_list).tolist(),
        labels=label_list,
        report_text=classification_report(y_true, y_pred, labels=label_list, zero_division=0),
    )


def select_best(results: dict[str, EvaluationResult]) -> str:
    """Escolhe o candidato por F1 macro; desempate pelo recall de ``urgente``."""
    if not results:
        raise ValueError("nenhum resultado para selecionar")
    return max(results, key=lambda k: (round(results[k].f1_macro, 4), results[k].urgent_recall))


def predict_proba_labels(pipe: Pipeline, texts: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Retorna (rotulos, probabilidades alinhadas a ``pipe.classes_``)."""
    normalized = prepare_texts(texts)
    proba = pipe.predict_proba(normalized)
    labels = pipe.classes_[np.argmax(proba, axis=1)]
    return labels, proba
