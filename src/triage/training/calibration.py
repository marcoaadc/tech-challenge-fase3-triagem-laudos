"""Avaliacao de calibracao das probabilidades (confianca vs. acuracia real).

A API devolve probabilidades por classe e a operacao clinica tende a usar a confianca como
criterio (ex.: encaminhar ``atencao`` de baixa confianca para revisao humana). Isso so faz
sentido se a probabilidade for calibrada: entre os laudos com confianca ~0,8, cerca de 80%
devem estar corretos. Este modulo mede isso sem alterar o modelo servido.

Metricas:
- **Brier score multiclasse**: media de ``sum_k (p_k - y_k)^2``; 0 e perfeito.
- **ECE (Expected Calibration Error)**: media ponderada de ``|acuracia - confianca|`` por faixa
  de confianca da classe prevista (top-label). Abaixo de 0,05 e considerado bem calibrado.
- **Diagrama de confiabilidade**: acuracia e confianca media por faixa, para o Model Card.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class ReliabilityBin:
    lower: float
    upper: float
    count: int
    mean_confidence: float
    accuracy: float


@dataclass
class CalibrationReport:
    n_samples: int
    brier_score: float
    ece: float
    mce: float
    mean_confidence: float
    accuracy: float
    bins: list[ReliabilityBin] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def calibration_report(
    proba: np.ndarray, y_true: Sequence[str], classes: Sequence[str], n_bins: int = 10
) -> CalibrationReport:
    """Calcula Brier, ECE/MCE e o diagrama de confiabilidade a partir das probabilidades previstas."""
    proba = np.asarray(proba, dtype=np.float64)
    classes = list(classes)
    if proba.ndim != 2 or proba.shape[1] != len(classes):
        raise ValueError(f"proba deve ter shape (n, {len(classes)}); recebido {proba.shape}")
    y_idx = np.array([classes.index(y) for y in y_true])
    if len(y_idx) != len(proba):
        raise ValueError("y_true e proba com tamanhos diferentes")

    one_hot = np.eye(len(classes))[y_idx]
    brier = float(np.mean(np.sum((proba - one_hot) ** 2, axis=1)))

    confidence = proba.max(axis=1)
    predicted = proba.argmax(axis=1)
    correct = (predicted == y_idx).astype(np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[ReliabilityBin] = []
    ece = 0.0
    mce = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (confidence > lo) & (confidence <= hi) if lo > 0 else (confidence >= lo) & (confidence <= hi)
        count = int(mask.sum())
        if count == 0:
            continue
        acc = float(correct[mask].mean())
        conf = float(confidence[mask].mean())
        gap = abs(acc - conf)
        ece += gap * count / len(proba)
        mce = max(mce, gap)
        bins.append(ReliabilityBin(lower=float(lo), upper=float(hi), count=count, mean_confidence=conf, accuracy=acc))

    return CalibrationReport(
        n_samples=int(len(proba)),
        brier_score=brier,
        ece=float(ece),
        mce=float(mce),
        mean_confidence=float(confidence.mean()),
        accuracy=float(correct.mean()),
        bins=bins,
    )


def render_reliability_table(report: CalibrationReport) -> str:
    lines = [
        "| Faixa de confianca | Amostras | Confianca media | Acuracia | Gap |",
        "|---|---|---|---|---|",
    ]
    for b in report.bins:
        lines.append(
            f"| {b.lower:.1f} - {b.upper:.1f} | {b.count} | {b.mean_confidence:.3f} | {b.accuracy:.3f} "
            f"| {abs(b.accuracy - b.mean_confidence):.3f} |"
        )
    return "\n".join(lines)
