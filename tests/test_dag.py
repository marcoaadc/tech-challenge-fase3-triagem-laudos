"""Teste de integridade da DAG do Airflow (executado apenas quando o Airflow esta instalado, ex.: job de CI)."""

from pathlib import Path

import pytest

# Importa "airflow.models" (e nao "airflow"): sem o Airflow instalado, a pasta airflow/ do projeto
# seria encontrada como pacote namespace e o skip nao aconteceria.
pytest.importorskip("airflow.models")

DAGS_DIR = Path(__file__).resolve().parents[1] / "airflow" / "dags"
EXPECTED_ORDER = [
    "ingest_data",
    "validate_data",
    "train_model",
    "export_onnx",
    "quality_gate",
    "promote_model",
    "benchmark_latency",
]


@pytest.fixture(scope="module")
def dagbag():
    from airflow.models import DagBag

    return DagBag(dag_folder=str(DAGS_DIR), include_examples=False)


def test_dag_loads_without_import_errors(dagbag) -> None:
    assert dagbag.import_errors == {}
    assert "triage_training_pipeline" in dagbag.dags


def test_dag_structure(dagbag) -> None:
    dag = dagbag.dags["triage_training_pipeline"]
    assert dag.task_ids and set(dag.task_ids) == set(EXPECTED_ORDER)
    assert dag.catchup is False
    assert dag.max_active_runs == 1
    for upstream, downstream in zip(EXPECTED_ORDER, EXPECTED_ORDER[1:], strict=True):
        assert downstream in dag.get_task(upstream).downstream_task_ids
