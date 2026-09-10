"""Validacao de esquema e qualidade do dataset antes do treino (gate do pipeline)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd

from triage.data.generator import LABELS

REQUIRED_COLUMNS = ("text", "label")


class DatasetValidationError(ValueError):
    """Dataset invalido para treino."""


@dataclass
class ValidationReport:
    n_rows: int
    n_duplicates: int
    class_distribution: dict[str, int]
    text_length_mean: float
    text_length_min: int
    text_length_max: int
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def validate_dataset(df: pd.DataFrame, min_rows: int = 2000, min_class_fraction: float = 0.05) -> ValidationReport:
    """Aplica checagens de contrato ao dataset. Levanta ``DatasetValidationError`` se reprovado.

    Regras (alinhadas ao enunciado: >= 2.000 amostras, colunas texto + target):
    - colunas obrigatorias presentes;
    - sem textos vazios/nulos;
    - rotulos dentro do conjunto conhecido;
    - todas as classes presentes com fracao minima (evita treinar sem a classe ``urgente``).
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DatasetValidationError(f"colunas obrigatorias ausentes: {missing}")
    if len(df) < min_rows:
        raise DatasetValidationError(f"dataset com {len(df)} linhas; minimo exigido e {min_rows}")

    texts = df["text"].astype("string")
    if texts.isna().any() or (texts.str.strip() == "").any():
        raise DatasetValidationError("existem textos nulos ou vazios")

    unknown = sorted(set(df["label"].unique()) - set(LABELS))
    if unknown:
        raise DatasetValidationError(f"rotulos desconhecidos: {unknown}; esperados {list(LABELS)}")

    distribution = {lb: int((df["label"] == lb).sum()) for lb in LABELS}
    warnings: list[str] = []
    for lb, count in distribution.items():
        frac = count / len(df)
        if count == 0:
            raise DatasetValidationError(f"classe '{lb}' ausente do dataset")
        if frac < min_class_fraction:
            raise DatasetValidationError(f"classe '{lb}' com fracao {frac:.3f} < minimo {min_class_fraction}")

    n_dup = int(texts.duplicated().sum())
    if n_dup > 0.10 * len(df):
        warnings.append(f"{n_dup} textos duplicados ({n_dup / len(df):.1%})")

    lengths = texts.str.len()
    return ValidationReport(
        n_rows=int(len(df)),
        n_duplicates=n_dup,
        class_distribution=distribution,
        text_length_mean=float(lengths.mean()),
        text_length_min=int(lengths.min()),
        text_length_max=int(lengths.max()),
        warnings=warnings,
    )
