from pathlib import Path

from triage.benchmark.latency import benchmark_predictor, render_markdown, run_benchmark
from triage.serving.predictor import OnnxPredictor


def test_benchmark_predictor_percentiles_are_ordered(trained_models_dir: Path, dataset_df) -> None:
    predictor = OnnxPredictor(trained_models_dir)
    stats = benchmark_predictor(predictor, dataset_df["text"].head(20).tolist(), n_calls=30, warmup=2)
    assert stats.backend == "onnx" and stats.mode == "single" and stats.n_calls == 30
    assert stats.min_ms <= stats.p50_ms <= stats.p95_ms <= stats.p99_ms <= stats.max_ms
    assert stats.throughput_per_s > 0


def test_run_benchmark_compares_backends(trained_models_dir: Path, dataset_df) -> None:
    report = run_benchmark(trained_models_dir, dataset_df["text"].head(40).tolist(), n_calls=40, batch_size=8)
    backends = {(r["backend"], r["mode"]) for r in report["results"]}
    assert backends == {("sklearn", "single"), ("sklearn", "batch"), ("onnx", "single"), ("onnx", "batch")}
    assert report["speedup_single_p50"] > 0
    markdown = render_markdown(report)
    assert "| onnx | single |" in markdown and "Speedup ONNX" in markdown
