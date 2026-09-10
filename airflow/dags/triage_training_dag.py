"""DAG de treino/retreino do classificador de triagem de laudos.

Fluxo (diario, sem catchup):

    ingest_data -> validate_data -> train_model -> export_onnx -> quality_gate -> promote_model -> benchmark_latency

- ``ingest_data``      : materializa o CSV de laudos (deterministico) em ``data/raw``.
- ``validate_data``    : gate de contrato do dataset (colunas, rotulos, volume minimo).
- ``train_model``      : treina os candidatos, seleciona por F1 macro e salva em ``models/candidate``.
- ``export_onnx``      : converte para ONNX e mede a paridade com o scikit-learn.
- ``quality_gate``     : branch: candidato aprovado -> promove; reprovado -> registra e encerra.
- ``promote_model``    : copia o candidato para ``models/`` (lido pela API) e atualiza o registry.
- ``benchmark_latency``: compara scikit-learn vs ONNX e grava ``reports/latency_benchmark.*``.

As tarefas chamam as funcoes ``run_*`` de ``triage.pipelines`` (testaveis sem o Airflow).
Os resultados intermediarios trafegam via XCom (apenas metadados pequenos, nunca artefatos).
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow.decorators import dag, task
from airflow.exceptions import AirflowSkipException

DATA_DIR = Path(os.environ.get("TRIAGE_DATA_DIR", "/opt/airflow/project/data"))
MODELS_DIR = Path(os.environ.get("TRIAGE_MODELS_DIR", "/opt/airflow/project/models"))
REPORTS_DIR = Path(os.environ.get("TRIAGE_REPORTS_DIR", "/opt/airflow/project/reports"))
MLFLOW_TRACKING_URI = os.environ.get("TRIAGE_MLFLOW_TRACKING_URI")

DATASET_PATH = DATA_DIR / "raw" / "laudos.csv"
CANDIDATE_DIR = MODELS_DIR / "candidate"

DEFAULT_ARGS = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
    "email_on_failure": False,
}


@dag(
    dag_id="triage_training_pipeline",
    description="Ingestao -> treino -> exportacao ONNX -> quality gate -> promocao do classificador de laudos",
    schedule="0 3 * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["mlops", "nlp", "triagem"],
    doc_md=__doc__,
)
def triage_training_pipeline():
    @task
    def ingest_data(n_samples: int = 6000, seed: int = 42) -> str:
        from triage.pipelines.ingest import run_ingest

        run_ingest(DATASET_PATH, n_samples=n_samples, seed=seed, reports_dir=REPORTS_DIR)
        return str(DATASET_PATH)

    @task
    def validate_data(dataset_path: str) -> dict:
        import pandas as pd

        from triage.data.validation import validate_dataset

        report = validate_dataset(pd.read_csv(dataset_path)).as_dict()
        return {"n_rows": report["n_rows"], "class_distribution": report["class_distribution"]}

    @task
    def train_model(dataset_path: str, validation: dict, seed: int = 42) -> dict:
        from triage.pipelines.train import run_train

        metadata = run_train(
            Path(dataset_path),
            CANDIDATE_DIR,
            seed=seed,
            reports_dir=REPORTS_DIR,
            mlflow_tracking_uri=MLFLOW_TRACKING_URI,
        )
        return {
            "model_version": metadata.model_version,
            "model_type": metadata.model_type,
            "f1_macro_test": metadata.metrics["test"]["f1_macro"],
            "urgent_recall_test": metadata.metrics["test"]["urgent_recall"],
            "dataset_rows": validation["n_rows"],
        }

    @task
    def export_onnx(training: dict) -> dict:
        from triage.pipelines.export import run_export

        parity = run_export(CANDIDATE_DIR, DATASET_PATH, parity_samples=1000, reports_dir=REPORTS_DIR)
        return {**training, "onnx_label_agreement": parity.get("label_agreement")}

    @task
    def quality_gate(export: dict) -> dict:
        from triage.pipelines.promote import check_gates
        from triage.training.metadata import ModelMetadata

        violations = check_gates(ModelMetadata.load(CANDIDATE_DIR))
        if violations:
            # Falha explicita: o modelo em producao permanece e o run fica vermelho para investigacao.
            raise AirflowSkipException(f"candidato reprovado no quality gate: {'; '.join(violations)}")
        return {**export, "gate": "approved"}

    @task
    def promote_model(gate: dict) -> dict:
        from triage.pipelines.promote import run_promote

        decision = run_promote(CANDIDATE_DIR, MODELS_DIR, reports_dir=REPORTS_DIR)
        return {**gate, "promoted": decision["promoted"]}

    @task
    def benchmark_latency(promotion: dict) -> dict:
        from triage.pipelines.benchmark import run_benchmark_stage

        report = run_benchmark_stage(MODELS_DIR, DATASET_PATH, REPORTS_DIR, n_calls=1000)
        return {**promotion, "speedup_single_p50": report["speedup_single_p50"]}

    dataset_path = ingest_data()
    validation = validate_data(dataset_path)
    training = train_model(dataset_path, validation)
    export = export_onnx(training)
    gate = quality_gate(export)
    promotion = promote_model(gate)
    benchmark_latency(promotion)


triage_training_pipeline()
