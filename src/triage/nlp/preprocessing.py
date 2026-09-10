"""Normalizacao de texto compartilhada entre treino e inferencia.

A mesma funcao e aplicada antes do TF-IDF no treino e antes da inferencia na API,
o que garante paridade entre o pipeline scikit-learn e o grafo ONNX exportado
(o tokenizador ONNX recebe texto ja em minusculas e sem acentos).
"""

from __future__ import annotations

import re
import unicodedata

_WHITESPACE_RE = re.compile(r"\s+")
_NUMBER_RE = re.compile(r"(\d+)[,.](\d+)")


def strip_accents(text: str) -> str:
    """Remove diacriticos (``pneumotórax`` -> ``pneumotorax``) preservando o restante."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def normalize_text(text: str) -> str:
    """Normaliza um laudo para o vocabulario do modelo.

    Passos (deterministicos e idempotentes):
    1. remove acentos e coloca em minusculas;
    2. unifica separador decimal (``13,8`` -> ``13.8``) para tokens numericos estaveis;
    3. colapsa espacos/quebras de linha.
    """
    if not text:
        return ""
    text = strip_accents(text).lower()
    text = _NUMBER_RE.sub(r"\1.\2", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text
