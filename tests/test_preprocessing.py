import pytest

from triage.nlp.preprocessing import normalize_text, strip_accents


def test_strip_accents_removes_diacritics() -> None:
    assert strip_accents("pneumotórax à esquerda, coração") == "pneumotorax a esquerda, coracao"


def test_normalize_lowercases_and_strips_accents() -> None:
    assert normalize_text("PNEUMOTÓRAX Volumoso") == "pneumotorax volumoso"


def test_normalize_unifies_decimal_separator() -> None:
    assert normalize_text("Hemoglobina: 13,8 g/dL") == "hemoglobina: 13.8 g/dl"


def test_normalize_collapses_whitespace() -> None:
    assert normalize_text("  Seios \n costofrênicos\t livres  ") == "seios costofrenicos livres"


@pytest.mark.parametrize("value", ["", None])
def test_normalize_empty_input(value) -> None:
    assert normalize_text(value) == ""


def test_normalize_is_idempotent() -> None:
    once = normalize_text("Área cardíaca: 12,5 cm")
    assert normalize_text(once) == once
