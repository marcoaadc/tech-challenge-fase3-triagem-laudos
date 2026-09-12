"""Estagio 2 - treino: seleciona o melhor candidato por validacao e salva o pipeline scikit-learn."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from triage import __version__
from triage.config.settings import get_settings
from triage.data.validation import validate_dataset
from triage.pipelines._cli import build_parser, setup_logging, write_json
from triage.training.calibration import calibration_report
from triage.training.metadata import SKLEARN_FILENAME, ModelMetadata, sha256_of
from triage.training.pipeline import (
    MODEL_CANDIDATES,
    TFIDF_PARAMS,
    build_pipeline,
    evaluate_pipeline,
    predict_proba_labels,
    select_best,
    train_pipeline,
)

logger = logging.getLogger("triage.pipelines.train")


def split_dataset(
    df: pd.DataFrame, seed: int, test_size: float = 0.15, val_size: float = 0.15
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split estratificado treino/validacao/teste (70/15/15 por padrao)."""
    train_val, test = train_test_split(df, test_size=test_size, stratify=df["label"], random_state=seed)
    rel_val = val_size / (1.0 - test_size)
    train, val = train_test_split(train_val, test_size=rel_val, stratify=train_val["label"], random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def _mlflow_log(
    tracking_uri: str, experiment: str, run_name: str, candidates: dict, best: str, metadata: ModelMetadata
) -> None:
    """Loga a busca de candidatos no MLflow (import tardio: dependencia opcional do grupo ``train``)."""
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({"tfidf_" + k: str(v) for k, v in TFIDF_PARAMS.items()})
        mlflow.log_param("seed", metadata.seed)
        mlflow.log_param("dataset_rows", metadata.dataset_rows)
        mlflow.set_tags({"best_model": best, "model_version": metadata.model_version})
        for name, info in candidates.items():
            with mlflow.start_run(run_name=name, nested=True):
                mlflow.log_param("model_type", name)
                mlflow.log_metrics(info["val"].flat_metrics("val_"))
                mlflow.log_metric("fit_seconds", info["fit_seconds"])
        mlflow.log_metrics({f"test_{k}": v for k, v in metadata.metrics["test"].items() if isinstance(v, float)})
        calibration = metadata.metrics.get("calibration", {})
        mlflow.log_metrics({f"test_{k}": calibration[k] for k in ("brier_score", "ece", "mce") if k in calibration})


def run_train(
    dataset_path: Path,
    output_dir: Path,
    seed: int = 42,
    candidates: list[str] | None = None,
    reports_dir: Path | None = None,
    mlflow_tracking_uri: str | None = None,
    mlflow_experiment: str = "triagem-laudos",
) -> ModelMetadata:
    """Treina os candidatos, escolhe o melhor por F1 macro na validacao e persiste em ``output_dir``."""
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    candidates = candidates or list(MODEL_CANDIDATES)

    df = pd.read_csv(dataset_path)
    validation = validate_dataset(df)
    train_df, val_df, test_df = split_dataset(df, seed=seed)
    logger.info("split: treino=%d validacao=%d teste=%d", len(train_df), len(val_df), len(test_df))

    results: dict[str, dict] = {}
    for name in candidates:
        pipe = build_pipeline(name, seed=seed)
        start = time.perf_counter()
        train_pipeline(pipe, train_df["text"], train_df["label"])
        fit_seconds = time.perf_counter() - start
        val_eval = evaluate_pipeline(pipe, val_df["text"], val_df["label"])
        results[name] = {"pipeline": pipe, "val": val_eval, "fit_seconds": fit_seconds}
        logger.info(
            "candidato %-14s f1_macro(val)=%.4f urgent_recall(val)=%.4f fit=%.1fs",
            name,
            val_eval.f1_macro,
            val_eval.urgent_recall,
            fit_seconds,
        )

    best = select_best({k: v["val"] for k, v in results.items()})
    best_pipe = results[best]["pipeline"]
    test_eval = evaluate_pipeline(best_pipe, test_df["text"], test_df["label"])
    logger.info("melhor candidato: %s | teste: f1_macro=%.4f acc=%.4f", best, test_eval.f1_macro, test_eval.accuracy)
    logger.info("\n%s", test_eval.report_text)
    _, test_proba = predict_proba_labels(best_pipe, test_df["text"])
    calibration = calibration_report(test_proba, test_df["label"], best_pipe.classes_)
    logger.info(
        "calibracao (teste): brier=%.4f ece=%.4f mce=%.4f", calibration.brier_score, calibration.ece, calibration.mce
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / SKLEARN_FILENAME
    joblib.dump(best_pipe, model_path, compress=3)

    dataset_sha = sha256_of(dataset_path)
    now = datetime.now(timezone.utc)
    clf = best_pipe.named_steps["clf"]
    metadata = ModelMetadata(
        model_version=f"{now:%Y%m%d.%H%M%S}-{dataset_sha[:8]}",
        model_type=f"tfidf+{best}",
        classes=[str(c) for c in best_pipe.classes_],
        trained_at=now.replace(microsecond=0).isoformat(),
        dataset_rows=int(len(df)),
        dataset_sha256=dataset_sha,
        hyperparameters={
            "tfidf": {k: (list(v) if isinstance(v, tuple) else v) for k, v in TFIDF_PARAMS.items()},
            "classifier": {k: v for k, v in clf.get_params().items() if isinstance(v, int | float | str | bool)},
            "vocabulary_size": int(len(best_pipe.named_steps["tfidf"].vocabulary_)),
        },
        metrics={
            "validation": {name: r["val"].as_dict() | {"fit_seconds": r["fit_seconds"]} for name, r in results.items()},
            "test": test_eval.as_dict(),
            "calibration": calibration.as_dict(),
            "dataset": validation.as_dict(),
        },
        seed=seed,
        package_version=__version__,
    )
    metadata.register_artifact("sklearn", model_path)
    metadata.save(output_dir)
    logger.info("modelo salvo em %s (versao %s)", output_dir, metadata.model_version)

    if reports_dir is not None:
        write_json(
            Path(reports_dir) / "training_metrics.json",
            {
                "model_version": metadata.model_version,
                "best_model": best,
                "validation": {
                    name: asdict(r["val"]) | {"fit_seconds": r["fit_seconds"]} for name, r in results.items()
                },
                "test": asdict(test_eval),
                "calibration": calibration.as_dict(),
            },
        )

    if mlflow_tracking_uri:
        try:
            run_name = f"train-{metadata.model_version}"
            _mlflow_log(mlflow_tracking_uri, mlflow_experiment, run_name, results, best, metadata)
            logger.info("run registrado no MLflow (%s)", mlflow_tracking_uri)
        except Exception as exc:  # MLflow e opcional: nao derruba o treino.
            logger.warning("falha ao logar no MLflow: %s", exc)

    return metadata


def main(argv: list[str] | None = None) -> None:
    settings = get_settings()
    parser = build_parser("Treina o classificador de urgencia e salva o pipeline scikit-learn.")
    parser.add_argument("--dataset", type=Path, default=settings.raw_dataset_path)
    parser.add_argument("--output-dir", type=Path, default=settings.models_dir / "candidate")
    parser.add_argument("--seed", type=int, default=settings.seed)
    parser.add_argument("--candidates", nargs="+", choices=sorted(MODEL_CANDIDATES), default=None)
    parser.add_argument("--reports-dir", type=Path, default=settings.reports_dir)
    parser.add_argument("--mlflow-tracking-uri", default=settings.mlflow_tracking_uri)
    parser.add_argument("--mlflow-experiment", default=settings.mlflow_experiment)
    args = parser.parse_args(argv)
    setup_logging(args.log_level)
    run_train(
        args.dataset,
        args.output_dir,
        seed=args.seed,
        candidates=args.candidates,
        reports_dir=args.reports_dir,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        mlflow_experiment=args.mlflow_experiment,
    )


if __name__ == "__main__":
    main()
