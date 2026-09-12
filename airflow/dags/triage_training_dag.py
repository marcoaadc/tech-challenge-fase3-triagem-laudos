"""DAG de treino/retreino do classificador de triagem de laudos.

Fluxo (diario, sem catchup):

    ingest_data -> validate_data -> train_model -> export_onnx -> quality_gate -> promote_model
                                                                      -> benchmark_latency -> reload_api

- ``ingest_data``      : materializa o CSV de laudos (deterministico) em ``data/raw``.
- ``validate_data``    : gate de contrato do dataset (colunas, rotulos, volume minimo).
- ``train_model``      : treina os candidatos, seleciona por F1 macro e salva em ``models/candidate``.
- ``export_onnx``      : converte para ONNX e mede a paridade com o scikit-learn.
- ``quality_gate``     : falha o run se o candidato nao atende aos criterios (modelo servido permanece).
- ``promote_model``    : copia o candidato para ``models/`` (lido pela API) e atualiza o registry.
- ``benchmark_latency``: compara scikit-learn vs ONNX e grava ``reports/latency_benchmark.*``.
- ``reload_api``       : chama ``POST /model/reload`` na API para servir o modelo novo sem restart.

As tarefas chamam as funcoes ``run_*`` de ``triage.pipelines`` (testaveis sem o Airflow).
Os resultados intermediarios trafegam via XCom (apenas metadados pequenos, nunca artefatos).
Ao final de cada run (sucesso ou falha) um resumo e gravado em ``reports/last_run.json``.

Parametros (trigger manual com "Trigger DAG w/ config"): ``n_samples``, ``seed``, ``min_f1_macro``,
``min_urgent_recall``.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from airflow.decorators import dag, task
from airflow.exceptions import AirflowFailException
from airflow.models.param import Param

DATA_DIR = Path(os.environ.get("TRIAGE_DATA_DIR", "/opt/airflow/project/data"))
MODELS_DIR = Path(os.environ.get("TRIAGE_MODELS_DIR", "/opt/airflow/project/models"))
REPORTS_DIR = Path(os.environ.get("TRIAGE_REPORTS_DIR", "/opt/airflow/project/reports"))
MLFLOW_TRACKING_URI = os.environ.get("TRIAGE_MLFLOW_TRACKING_URI")
API_URL = os.environ.get("TRIAGE_API_URL", "")  # vazio desativa o reload
API_KEY = os.environ.get("TRIAGE_API_KEY", "")

DATASET_PATH = DATA_DIR / "raw" / "laudos.csv"
CANDIDATE_DIR = MODELS_DIR / "candidate"

DEFAULT_ARGS = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
    "execution_timeout": timedelta(minutes=20),
    "email_on_failure": False,
}


def _write_last_run(context: dict, status: str) -> None:
    """Callback de DAG: grava um resumo do run para consulta rapida (e para o dashboard/alertas)."""
    dag_run = context.get("dag_run")
    ti_states = {}
    if dag_run is not None:
        ti_states = {ti.task_id: ti.state for ti in dag_run.get_task_instances()}
    promotion_path = REPORTS_DIR / "promotion.json"
    promotion = json.loads(promotion_path.read_text(encoding="utf-8")) if promotion_path.exists() else {}
    summary = {
        "dag_id": context["dag"].dag_id,
        "run_id": getattr(dag_run, "run_id", None),
        "status": status,
        "finished_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "tasks": ti_states,
        "model_version": promotion.get("model_version"),
        "promoted": promotion.get("promoted"),
        "violations": promotion.get("violations", []),
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "last_run.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def on_success(context: dict) -> None:
    _write_last_run(context, "success")


def on_failure(context: dict) -> None:
    _write_last_run(context, "failed")


@dag(
    dag_id="triage_training_pipeline",
    description="Ingestao -> treino -> exportacao ONNX -> quality gate -> promocao -> reload da API",
    schedule="0 3 * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    dagrun_timeout=timedelta(hours=1),
    on_success_callback=on_success,
    on_failure_callback=on_failure,
    params={
        "n_samples": Param(6000, type="integer", minimum=2000, description="Laudos gerados na ingestao"),
        "seed": Param(42, type="integer", description="Seed do gerador, do split e dos modelos"),
        "min_f1_macro": Param(0.90, type="number", minimum=0, maximum=1, description="Gate: F1 macro minimo"),
        "min_urgent_recall": Param(0.90, type="number", minimum=0, maximum=1, description="Gate: recall urgente"),
    },
    tags=["mlops", "nlp", "triagem"],
    doc_md=__doc__,
)
def triage_training_pipeline():
    @task
    def ingest_data(**context) -> str:
        from triage.pipelines.ingest import run_ingest

        params = context["params"]
        run_ingest(DATASET_PATH, n_samples=int(params["n_samples"]), seed=int(params["seed"]), reports_dir=REPORTS_DIR)
        return str(DATASET_PATH)

    @task
    def validate_data(dataset_path: str) -> dict:
        import pandas as pd

        from triage.data.validation import validate_dataset

        report = validate_dataset(pd.read_csv(dataset_path)).as_dict()
        return {"n_rows": report["n_rows"], "class_distribution": report["class_distribution"]}

    @task
    def train_model(dataset_path: str, validation: dict, **context) -> dict:
        from triage.pipelines.train import run_train

        metadata = run_train(
            Path(dataset_path),
            CANDIDATE_DIR,
            seed=int(context["params"]["seed"]),
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
    def quality_gate(export: dict, **context) -> dict:
        from triage.pipelines.promote import Gates, check_gates
        from triage.training.metadata import ModelMetadata

        params = context["params"]
        gates = Gates(min_f1_macro=float(params["min_f1_macro"]), min_urgent_recall=float(params["min_urgent_recall"]))
        violations = check_gates(ModelMetadata.load(CANDIDATE_DIR), gates)
        if violations:
            # Falha explicita (run vermelho): o modelo em producao permanece e o motivo fica no log.
            raise AirflowFailException(f"candidato reprovado no quality gate: {'; '.join(violations)}")
        return {**export, "gate": "approved", "gates": gates.__dict__}

    @task
    def promote_model(gate: dict) -> dict:
        from triage.pipelines.promote import Gates, run_promote

        decision = run_promote(CANDIDATE_DIR, MODELS_DIR, Gates(**gate["gates"]), reports_dir=REPORTS_DIR)
        return {**gate, "promoted": decision["promoted"]}

    @task
    def benchmark_latency(promotion: dict) -> dict:
        from triage.pipelines.benchmark import run_benchmark_stage

        report = run_benchmark_stage(MODELS_DIR, DATASET_PATH, REPORTS_DIR, n_calls=1000)
        return {**promotion, "speedup_single_p50": report["speedup_single_p50"]}

    @task(retries=3, retry_delay=timedelta(seconds=30))
    def reload_api(benchmark: dict) -> dict:
        """Pede a API para servir o modelo recem-promovido. Sem TRIAGE_API_URL, apenas registra."""
        import urllib.request

        if not API_URL:
            return {**benchmark, "api_reload": "skipped (TRIAGE_API_URL nao definida)"}
        request = urllib.request.Request(API_URL.rstrip("/") + "/model/reload", method="POST")
        if API_KEY:
            request.add_header("X-API-Key", API_KEY)
        with urllib.request.urlopen(request, timeout=30) as response:
            body = json.loads(response.read().decode("utf-8"))
        if body.get("versao_atual") != benchmark["model_version"]:
            raise AirflowFailException(
                f"API serve {body.get('versao_atual')} mas o modelo promovido e {benchmark['model_version']}"
            )
        return {**benchmark, "api_reload": body}

    dataset_path = ingest_data()
    validation = validate_data(dataset_path)
    training = train_model(dataset_path, validation)
    export = export_onnx(training)
    gate = quality_gate(export)
    promotion = promote_model(gate)
    benchmark = benchmark_latency(promotion)
    reload_api(benchmark)


triage_training_pipeline()
