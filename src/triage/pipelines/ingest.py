"""Estagio 1 - ingestao: materializa o dataset de laudos e valida seu contrato."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from triage.config.settings import get_settings
from triage.data.generator import generate_dataset, write_csv
from triage.data.validation import validate_dataset
from triage.pipelines._cli import build_parser, setup_logging, write_json

logger = logging.getLogger("triage.pipelines.ingest")


def run_ingest(output_path: Path, n_samples: int = 6000, seed: int = 42, reports_dir: Path | None = None) -> dict:
    """Gera o CSV de laudos (deterministico por ``seed``) e grava o relatorio de validacao."""
    output_path = Path(output_path)
    reports = generate_dataset(n_samples=n_samples, seed=seed)
    write_csv(reports, output_path)
    logger.info("dataset gerado: %s (%d laudos)", output_path, len(reports))

    df = pd.read_csv(output_path)
    validation = validate_dataset(df).as_dict()
    logger.info("validacao ok: distribuicao=%s", validation["class_distribution"])
    if reports_dir is not None:
        write_json(Path(reports_dir) / "dataset_validation.json", validation)
    return validation


def main(argv: list[str] | None = None) -> None:
    settings = get_settings()
    parser = build_parser("Gera e valida o dataset de laudos.")
    parser.add_argument("--output", type=Path, default=settings.raw_dataset_path)
    parser.add_argument("--n-samples", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=settings.seed)
    parser.add_argument("--reports-dir", type=Path, default=settings.reports_dir)
    args = parser.parse_args(argv)
    setup_logging(args.log_level)
    run_ingest(args.output, n_samples=args.n_samples, seed=args.seed, reports_dir=args.reports_dir)


if __name__ == "__main__":
    main()
