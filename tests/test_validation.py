import pandas as pd
import pytest

from triage.data.validation import DatasetValidationError, validate_dataset


def _df(n: int = 30, labels=("normal", "atencao", "urgente")) -> pd.DataFrame:
    texts = [f"laudo numero {i}" for i in range(n)]
    return pd.DataFrame({"text": texts, "label": [labels[i % len(labels)] for i in range(n)]})


def test_valid_dataset_returns_report() -> None:
    report = validate_dataset(_df(), min_rows=10)
    assert report.n_rows == 30
    assert report.class_distribution == {"normal": 10, "atencao": 10, "urgente": 10}
    assert report.text_length_min > 0
    assert report.as_dict()["n_duplicates"] == 0


def test_missing_columns() -> None:
    with pytest.raises(DatasetValidationError, match="colunas obrigatorias"):
        validate_dataset(pd.DataFrame({"texto": ["a"], "label": ["normal"]}), min_rows=1)


def test_too_few_rows() -> None:
    with pytest.raises(DatasetValidationError, match="minimo exigido"):
        validate_dataset(_df(), min_rows=100)


def test_empty_text_rejected() -> None:
    df = _df()
    df.loc[0, "text"] = "   "
    with pytest.raises(DatasetValidationError, match="vazios"):
        validate_dataset(df, min_rows=10)


def test_unknown_label_rejected() -> None:
    df = _df()
    df.loc[0, "label"] = "critico"
    with pytest.raises(DatasetValidationError, match="rotulos desconhecidos"):
        validate_dataset(df, min_rows=10)


def test_missing_class_rejected() -> None:
    with pytest.raises(DatasetValidationError, match="ausente"):
        validate_dataset(_df(labels=("normal", "atencao")), min_rows=10)


def test_warns_on_many_duplicates() -> None:
    df = pd.DataFrame({"text": ["mesmo laudo"] * 30, "label": ["normal", "atencao", "urgente"] * 10})
    report = validate_dataset(df, min_rows=10)
    assert report.warnings and "duplicados" in report.warnings[0]
