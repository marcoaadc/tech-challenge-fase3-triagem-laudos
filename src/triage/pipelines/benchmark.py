"""Estagio 5 - benchmark: compara a latencia do modelo original (scikit-learn) e do otimizado (ONNX)."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from triage.benchmark.latency import render_markdown, run_benchmark
from triage.config.settings import get_settings
from triage.pipelines._cli import build_parser, setup_logging, write_json

logger = logging.getLogger("triage.pipelines.benchmark")


def run_benchmark_stage(
    models_dir: Path, dataset_path: Path, reports_dir: Path, n_calls: int = 1000, n_texts: int = 500
) -> dict:
    df = pd.read_csv(dataset_path)
    texts = df["text"].sample(n=min(n_texts, len(df)), random_state=0).tolist()
    report = run_benchmark(models_dir, texts, n_calls=n_calls)
    reports_dir = Path(reports_dir)
    write_json(reports_dir / "latency_benchmark.json", report)
    (reports_dir / "latency_benchmark.md").write_text(render_markdown(report), encoding="utf-8")
    for r in report["results"]:
        logger.info(
            "%-8s %-6s batch=%-3d p50=%.3fms p95=%.3fms p99=%.3fms thr=%.0f/s",
            r["backend"],
            r["mode"],
            r["batch_size"],
            r["p50_ms"],
            r["p95_ms"],
            r["p99_ms"],
            r["throughput_per_s"],
        )
    logger.info(
        "speedup ONNX (unitario): %.1fx p50 | %.1fx p95", report["speedup_single_p50"], report["speedup_single_p95"]
    )
    return report


def main(argv: list[str] | None = None) -> None:
    settings = get_settings()
    parser = build_parser("Benchmark de latencia scikit-learn vs ONNX Runtime.")
    parser.add_argument("--models-dir", type=Path, default=settings.models_dir)
    parser.add_argument("--dataset", type=Path, default=settings.raw_dataset_path)
    parser.add_argument("--reports-dir", type=Path, default=settings.reports_dir)
    parser.add_argument("--n-calls", type=int, default=1000)
    args = parser.parse_args(argv)
    setup_logging(args.log_level)
    run_benchmark_stage(args.models_dir, args.dataset, args.reports_dir, n_calls=args.n_calls)


if __name__ == "__main__":
    main()
