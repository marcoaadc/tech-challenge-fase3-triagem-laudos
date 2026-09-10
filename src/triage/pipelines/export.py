"""Estagio 3 - exportacao: converte o pipeline treinado para ONNX e verifica a paridade."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd

from triage.config.settings import get_settings
from triage.nlp.preprocessing import normalize_text
from triage.pipelines._cli import build_parser, setup_logging, write_json
from triage.training.export import check_parity, export_onnx
from triage.training.metadata import ONNX_FILENAME, SKLEARN_FILENAME, ModelMetadata

logger = logging.getLogger("triage.pipelines.export")


def run_export(
    models_dir: Path, dataset_path: Path | None = None, parity_samples: int = 1000, reports_dir: Path | None = None
) -> dict:
    models_dir = Path(models_dir)
    metadata = ModelMetadata.load(models_dir)
    pipe = joblib.load(models_dir / SKLEARN_FILENAME)

    onnx_path = export_onnx(pipe, models_dir / ONNX_FILENAME)
    logger.info("modelo ONNX exportado: %s (%d bytes)", onnx_path, onnx_path.stat().st_size)

    parity: dict = {}
    if dataset_path is not None and Path(dataset_path).exists():
        df = pd.read_csv(dataset_path)
        sample = df.sample(n=min(parity_samples, len(df)), random_state=metadata.seed)
        normalized = [normalize_text(t) for t in sample["text"]]
        report = check_parity(pipe, onnx_path, normalized)
        parity = report.as_dict()
        logger.info(
            "paridade sklearn x onnx: rotulos=%.4f | prob diff max=%.5f media=%.6f",
            report.label_agreement,
            report.max_abs_prob_diff,
            report.mean_abs_prob_diff,
        )

    metadata.register_artifact("onnx", onnx_path)
    metadata.metrics["onnx_parity"] = parity
    metadata.save(models_dir)
    if reports_dir is not None:
        write_json(Path(reports_dir) / "onnx_parity.json", {"model_version": metadata.model_version, **parity})
    return parity


def main(argv: list[str] | None = None) -> None:
    settings = get_settings()
    parser = build_parser("Exporta o pipeline treinado para ONNX e verifica paridade.")
    parser.add_argument("--models-dir", type=Path, default=settings.models_dir / "candidate")
    parser.add_argument("--dataset", type=Path, default=settings.raw_dataset_path)
    parser.add_argument("--parity-samples", type=int, default=1000)
    parser.add_argument("--reports-dir", type=Path, default=settings.reports_dir)
    args = parser.parse_args(argv)
    setup_logging(args.log_level)
    run_export(args.models_dir, args.dataset, args.parity_samples, args.reports_dir)


if __name__ == "__main__":
    main()
