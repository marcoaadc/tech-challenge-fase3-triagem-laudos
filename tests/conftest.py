"""Fixtures compartilhadas.

``trained_models_dir`` treina um modelo real (pequeno) uma unica vez por sessao de testes e o
exporta para ONNX, permitindo testar predictor, API e benchmark contra artefatos verdadeiros.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from triage.config.settings import Settings
from triage.data.generator import generate_dataset, write_csv
from triage.pipelines.export import run_export
from triage.pipelines.promote import Gates, run_promote
from triage.pipelines.train import run_train

URGENT_TEXT = (
    "TOMOGRAFIA COMPUTADORIZADA DE CRÂNIO SEM CONTRASTE. Paciente masculino, 71 anos. "
    "Hematoma subdural agudo fronto-parietal direito com desvio de linha média de 9 mm. "
    "ACHADO CRÍTICO - comunicado ao médico solicitante."
)
NORMAL_TEXT = (
    "RADIOGRAFIA DE TÓRAX EM PA E PERFIL. Campos pulmonares com transparência preservada. "
    "Seios costofrênicos livres. Área cardíaca dentro dos limites da normalidade. Exame normal."
)
ATTENTION_TEXT = (
    "EXAMES LABORATORIAIS. Hemoglobina: 10,9 g/dL, anemia leve normocítica. "
    "Glicemia de jejum: 131 mg/dL. Sugere-se correlação clínico-laboratorial e reavaliação em 30 dias."
)


@pytest.fixture(scope="session")
def dataset_csv(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("data") / "laudos.csv"
    write_csv(generate_dataset(n_samples=2400, seed=7), path)
    return path


@pytest.fixture(scope="session")
def dataset_df(dataset_csv: Path) -> pd.DataFrame:
    return pd.read_csv(dataset_csv)


@pytest.fixture(scope="session")
def trained_models_dir(tmp_path_factory: pytest.TempPathFactory, dataset_csv: Path) -> Path:
    candidate = tmp_path_factory.mktemp("models") / "candidate"
    target = candidate.parent / "production"
    run_train(dataset_csv, candidate, seed=7, candidates=["logreg"])
    run_export(candidate, dataset_csv, parity_samples=300)
    # Modelo pequeno (2.400 amostras): gates relaxados em relacao aos de producao.
    run_promote(candidate, target, Gates(min_f1_macro=0.80, min_urgent_recall=0.80))
    return target


@pytest.fixture
def settings_onnx(trained_models_dir: Path) -> Settings:
    return Settings(models_dir=trained_models_dir, model_backend="onnx", _env_file=None)


@pytest.fixture
def settings_sklearn(trained_models_dir: Path) -> Settings:
    return Settings(models_dir=trained_models_dir, model_backend="sklearn", _env_file=None)
