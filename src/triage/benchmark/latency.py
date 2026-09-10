"""Benchmark de latencia in-process: pipeline scikit-learn original vs. modelo ONNX Runtime.

Mede a latencia de inferencia *unitaria* (1 laudo por chamada, cenario da API em tempo real)
e em *lote*, reportando percentis. A comparacao HTTP ponta a ponta fica em ``scripts/load_test.py``.
"""

from __future__ import annotations

import platform
import statistics
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from triage.serving.predictor import OnnxPredictor, Predictor, SklearnPredictor
from triage.training.metadata import ONNX_FILENAME, SKLEARN_FILENAME, ModelMetadata


@dataclass
class LatencyStats:
    backend: str
    mode: str
    n_calls: int
    batch_size: int
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float
    throughput_per_s: float

    def as_dict(self) -> dict:
        return asdict(self)


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * pct
    lo, hi = int(k), min(int(k) + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (k - lo)


def benchmark_predictor(
    predictor: Predictor, texts: Sequence[str], n_calls: int = 1000, batch_size: int = 1, warmup: int = 20
) -> LatencyStats:
    texts = list(texts)
    if not texts:
        raise ValueError("texts vazio")
    for i in range(warmup):
        predictor.predict([texts[i % len(texts)]])

    samples_ms: list[float] = []
    total_items = 0
    start_all = time.perf_counter()
    for i in range(n_calls):
        if batch_size == 1:
            batch = [texts[i % len(texts)]]
        else:
            offset = (i * batch_size) % len(texts)
            batch = [texts[(offset + j) % len(texts)] for j in range(batch_size)]
        start = time.perf_counter()
        predictor.predict(batch)
        samples_ms.append((time.perf_counter() - start) * 1000.0)
        total_items += len(batch)
    wall = time.perf_counter() - start_all

    samples_ms.sort()
    return LatencyStats(
        backend=predictor.backend,
        mode="single" if batch_size == 1 else "batch",
        n_calls=n_calls,
        batch_size=batch_size,
        p50_ms=_percentile(samples_ms, 0.50),
        p90_ms=_percentile(samples_ms, 0.90),
        p95_ms=_percentile(samples_ms, 0.95),
        p99_ms=_percentile(samples_ms, 0.99),
        mean_ms=statistics.fmean(samples_ms),
        min_ms=samples_ms[0],
        max_ms=samples_ms[-1],
        throughput_per_s=total_items / wall if wall > 0 else 0.0,
    )


def run_benchmark(models_dir: Path, texts: Sequence[str], n_calls: int = 1000, batch_size: int = 32) -> dict:
    models_dir = Path(models_dir)
    metadata = ModelMetadata.load(models_dir)
    predictors: list[Predictor] = [SklearnPredictor(models_dir, metadata), OnnxPredictor(models_dir, metadata)]

    results: list[LatencyStats] = []
    for predictor in predictors:
        results.append(benchmark_predictor(predictor, texts, n_calls=n_calls, batch_size=1))
        n_batches = max(n_calls // batch_size, 20)
        results.append(benchmark_predictor(predictor, texts, n_calls=n_batches, batch_size=batch_size))

    single = {r.backend: r for r in results if r.mode == "single"}
    speedup_p50 = single["sklearn"].p50_ms / single["onnx"].p50_ms if single["onnx"].p50_ms else 0.0
    speedup_p95 = single["sklearn"].p95_ms / single["onnx"].p95_ms if single["onnx"].p95_ms else 0.0

    return {
        "model_version": metadata.model_version,
        "model_type": metadata.model_type,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
        },
        "artifacts": {
            "sklearn_bytes": (models_dir / SKLEARN_FILENAME).stat().st_size,
            "onnx_bytes": (models_dir / ONNX_FILENAME).stat().st_size,
        },
        "n_texts": len(texts),
        "results": [r.as_dict() for r in results],
        "speedup_single_p50": speedup_p50,
        "speedup_single_p95": speedup_p95,
    }


def render_markdown(report: dict) -> str:
    lines = [
        f"# Benchmark de latencia - modelo {report['model_version']} ({report['model_type']})",
        "",
        f"Ambiente: Python {report['environment']['python']} | {report['environment']['platform']} | "
        f"{report['environment']['processor']}",
        "",
        f"Tamanho dos artefatos: scikit-learn {report['artifacts']['sklearn_bytes'] / 1024:.0f} KB | "
        f"ONNX {report['artifacts']['onnx_bytes'] / 1024:.0f} KB",
        "",
        "| Backend | Modo | Batch | Chamadas | p50 (ms) | p90 (ms) | p95 (ms) | p99 (ms) | Media (ms) "
        "| Throughput (laudos/s) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in report["results"]:
        lines.append(
            f"| {r['backend']} | {r['mode']} | {r['batch_size']} | {r['n_calls']} | {r['p50_ms']:.3f} "
            f"| {r['p90_ms']:.3f} | {r['p95_ms']:.3f} | {r['p99_ms']:.3f} | {r['mean_ms']:.3f} "
            f"| {r['throughput_per_s']:.0f} |"
        )
    lines += [
        "",
        f"**Speedup ONNX vs scikit-learn (inferencia unitaria):** {report['speedup_single_p50']:.1f}x no p50, "
        f"{report['speedup_single_p95']:.1f}x no p95.",
        "",
    ]
    return "\n".join(lines)
