import json
import logging
from pathlib import Path

import pytest

from triage.api.logging_config import JsonFormatter, TextFormatter, configure_logging
from triage.pipelines.train import run_train


def _record(**extra) -> logging.LogRecord:
    record = logging.LogRecord("triage.test", logging.INFO, __file__, 10, "mensagem %s", ("x",), None)
    for k, v in extra.items():
        setattr(record, k, v)
    return record


def test_json_formatter_includes_extras_and_message() -> None:
    payload = json.loads(JsonFormatter().format(_record(request_id="abc", duration_ms=1.5)))
    assert payload["message"] == "mensagem x"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "triage.test"
    assert payload["request_id"] == "abc" and payload["duration_ms"] == 1.5
    assert payload["timestamp"].endswith("+00:00")


def test_json_formatter_serializes_exception() -> None:
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        import sys

        record = _record()
        record.exc_info = sys.exc_info()
    payload = json.loads(JsonFormatter().format(record))
    assert "RuntimeError: boom" in payload["exception"]


def test_text_formatter_appends_extras() -> None:
    line = TextFormatter("%(levelname)s %(message)s").format(_record(path="/predict"))
    assert line.startswith("INFO mensagem x") and "path=/predict" in line


def test_configure_logging_replaces_root_handlers() -> None:
    configure_logging("DEBUG", json_logs=True)
    root = logging.getLogger()
    assert root.level == logging.DEBUG
    assert len(root.handlers) == 1 and isinstance(root.handlers[0].formatter, JsonFormatter)
    assert logging.getLogger("uvicorn.access").disabled is True
    configure_logging("INFO", json_logs=False)
    assert isinstance(logging.getLogger().handlers[0].formatter, TextFormatter)


@pytest.mark.slow
def test_train_logs_to_mlflow(dataset_csv: Path, tmp_path: Path) -> None:
    mlflow = pytest.importorskip("mlflow")
    tracking = (tmp_path / "mlruns").as_uri()
    metadata = run_train(
        dataset_csv,
        tmp_path / "candidate",
        seed=3,
        candidates=["logreg", "complement_nb"],
        mlflow_tracking_uri=tracking,
        mlflow_experiment="teste-triagem",
    )
    mlflow.set_tracking_uri(tracking)
    experiment = mlflow.get_experiment_by_name("teste-triagem")
    assert experiment is not None
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], output_format="list")
    parents = [r for r in runs if "mlflow.parentRunId" not in r.data.tags]
    children = [r for r in runs if "mlflow.parentRunId" in r.data.tags]
    assert len(parents) == 1 and len(children) == 2
    parent = parents[0]
    assert parent.data.tags["model_version"] == metadata.model_version
    assert parent.data.tags["best_model"] in {"logreg", "complement_nb"}
    assert parent.data.metrics["test_f1_macro"] == pytest.approx(metadata.metrics["test"]["f1_macro"])
    assert "test_ece" in parent.data.metrics
    assert {c.data.params["model_type"] for c in children} == {"logreg", "complement_nb"}
    assert all("val_f1_macro" in c.data.metrics for c in children)
