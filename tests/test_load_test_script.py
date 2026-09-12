"""Testa o script de carga contra uma API real (uvicorn em thread, porta livre)."""

import importlib.util
import json
import socket
import threading
import time
from pathlib import Path

import pytest
import uvicorn

from triage.api.main import create_app
from triage.config.settings import Settings

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "load_test.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("load_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def live_api(trained_models_dir: Path):
    port = _free_port()
    app = create_app(Settings(models_dir=trained_models_dir, _env_file=None))
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 15
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    assert server.started, "uvicorn nao subiu"
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=10)


def test_load_tester_fixed_mode(live_api: str) -> None:
    lt = _load_script()
    tester = lt.LoadTester(live_api, lt.SAMPLE_TEXTS)
    wall = tester.run_fixed(n_requests=30, concurrency=3)
    summary = tester.summary(wall)
    assert summary["total_requests"] == 30
    assert summary["successful"] == 30 and summary["errors"] == 0
    assert summary["statuses"] == {200: 30}
    assert summary["http_latency_ms"]["p50"] > 0
    assert summary["model_latency_ms"]["p50"] > 0
    assert summary["achieved_rps"] > 0


def test_load_tester_counts_errors(live_api: str) -> None:
    lt = _load_script()
    tester = lt.LoadTester(live_api, [""])  # texto vazio -> 422
    tester.run_fixed(n_requests=5, concurrency=1)
    summary = tester.summary(1.0)
    assert summary["errors"] == 5 and summary["statuses"] == {422: 5}
    assert summary["error_rate"] == 1.0


def test_main_writes_summary_and_exit_code(live_api: str, tmp_path: Path) -> None:
    lt = _load_script()
    output = tmp_path / "out" / "summary.json"
    code = lt.main(
        ["--url", live_api, "--requests", "10", "--concurrency", "2", "--output", str(output), "--label", "t"]
    )
    assert code == 0
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["label"] == "t" and saved["total_requests"] == 10


def test_percentile_helper() -> None:
    lt = _load_script()
    assert lt.percentile([], 0.5) == 0.0
    assert lt.percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert lt.percentile([5.0], 0.99) == 5.0
