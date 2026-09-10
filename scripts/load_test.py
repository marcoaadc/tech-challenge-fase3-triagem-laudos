"""Teste de carga HTTP contra a API de triagem (somente biblioteca padrao).

Mede a latencia ponta a ponta (cliente -> API -> modelo) e serve como gerador de trafego
para popular os paineis do Grafana. Exemplos:

    python scripts/load_test.py --url http://127.0.0.1:8000 --requests 2000 --concurrency 8
    python scripts/load_test.py --url http://127.0.0.1:8000 --duration 300 --rps 20
    python scripts/load_test.py --output reports/load_test_onnx.json

Use 127.0.0.1 em vez de localhost: no Windows, "localhost" resolve primeiro para IPv6 (::1) e cada
requisicao perde ~2 s ate cair para IPv4 quando o servidor escuta apenas em 127.0.0.1.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SAMPLE_TEXTS = [
    "RADIOGRAFIA DE TÓRAX EM PA E PERFIL. Campos pulmonares com transparência preservada. "
    "Seios costofrênicos livres. Exame normal.",
    "EXAMES LABORATORIAIS. Hemoglobina: 10,9 g/dL, anemia leve normocítica. Glicemia de jejum: 131 mg/dL. "
    "Sugere-se correlação clínico-laboratorial.",
    "TC de crânio sem contraste. Hematoma subdural agudo fronto-parietal direito com desvio de linha média de 9 mm. "
    "ACHADO CRÍTICO.",
    "ELETROCARDIOGRAMA DE 12 DERIVAÇÕES. Ritmo sinusal, FC 72 bpm. Sem alterações da repolarização ventricular.",
    "NOTA DE TRIAGEM. Dor torácica opressiva há 40 minutos com irradiação para MSE e sudorese fria. "
    "SatO2 84% em ar ambiente.",
    "ULTRASSONOGRAFIA DE ABDOME TOTAL. Esteatose hepática grau II. Colelitíase sem sinais de colecistite.",
]


def load_texts(dataset: Path | None, limit: int = 500) -> list[str]:
    if dataset and dataset.exists():
        with dataset.open(encoding="utf-8", newline="") as fh:
            rows = [row["text"] for row in csv.DictReader(fh)]
        random.Random(0).shuffle(rows)
        return rows[:limit] or SAMPLE_TEXTS
    return SAMPLE_TEXTS


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


class LoadTester:
    def __init__(self, url: str, texts: list[str], timeout: float = 5.0) -> None:
        self.endpoint = url.rstrip("/") + "/predict"
        self.texts = texts
        self.timeout = timeout
        self.latencies_ms: list[float] = []
        self.model_ms: list[float] = []
        self.errors = 0
        self.statuses: dict[int, int] = {}
        self._lock = threading.Lock()

    def one_request(self, i: int) -> None:
        payload = json.dumps({"texto": self.texts[i % len(self.texts)]}).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint, data=payload, headers={"Content-Type": "application/json"}, method="POST"
        )
        start = time.perf_counter()
        status, model_ms = 0, None
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                status = response.status
                body = json.loads(response.read().decode("utf-8"))
                model_ms = body.get("latencia_ms")
        except urllib.error.HTTPError as exc:
            status = exc.code
        except Exception:
            status = 0
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        with self._lock:
            self.statuses[status] = self.statuses.get(status, 0) + 1
            if status == 200:
                self.latencies_ms.append(elapsed_ms)
                if model_ms is not None:
                    self.model_ms.append(float(model_ms))
            else:
                self.errors += 1

    def run_fixed(self, n_requests: int, concurrency: int) -> float:
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            list(pool.map(self.one_request, range(n_requests)))
        return time.perf_counter() - start

    def run_paced(self, duration_s: float, rps: float, concurrency: int) -> float:
        interval = 1.0 / rps
        start = time.perf_counter()
        i = 0
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            while time.perf_counter() - start < duration_s:
                pool.submit(self.one_request, i)
                i += 1
                sleep_for = start + i * interval - time.perf_counter()
                if sleep_for > 0:
                    time.sleep(sleep_for)
        return time.perf_counter() - start

    def summary(self, wall_s: float) -> dict:
        total = sum(self.statuses.values())
        lat = self.latencies_ms
        return {
            "endpoint": self.endpoint,
            "total_requests": total,
            "successful": len(lat),
            "errors": self.errors,
            "error_rate": (self.errors / total) if total else 0.0,
            "statuses": self.statuses,
            "wall_seconds": wall_s,
            "achieved_rps": (total / wall_s) if wall_s else 0.0,
            "http_latency_ms": {
                "p50": percentile(lat, 0.50),
                "p90": percentile(lat, 0.90),
                "p95": percentile(lat, 0.95),
                "p99": percentile(lat, 0.99),
                "mean": statistics.fmean(lat) if lat else 0.0,
                "max": max(lat) if lat else 0.0,
            },
            "model_latency_ms": {
                "p50": percentile(self.model_ms, 0.50),
                "p95": percentile(self.model_ms, 0.95),
                "p99": percentile(self.model_ms, 0.99),
            },
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Teste de carga da API de triagem.")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--dataset", type=Path, default=Path("data/raw/laudos.csv"))
    parser.add_argument("--requests", type=int, default=1000, help="modo fixo: numero total de requisicoes")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--duration", type=float, default=None, help="modo continuo: duracao em segundos")
    parser.add_argument("--rps", type=float, default=10.0, help="modo continuo: requisicoes por segundo")
    parser.add_argument("--output", type=Path, default=None, help="salva o resumo em JSON")
    parser.add_argument("--label", default=None, help="rotulo livre gravado no resumo (ex.: backend testado)")
    args = parser.parse_args(argv)

    tester = LoadTester(args.url, load_texts(args.dataset))
    if args.duration:
        print(f"modo continuo: {args.duration:.0f}s a {args.rps:.0f} req/s -> {tester.endpoint}")
        wall = tester.run_paced(args.duration, args.rps, args.concurrency)
    else:
        print(f"modo fixo: {args.requests} requisicoes, concorrencia {args.concurrency} -> {tester.endpoint}")
        wall = tester.run_fixed(args.requests, args.concurrency)

    summary = tester.summary(wall)
    if args.label:
        summary["label"] = args.label
    http, model = summary["http_latency_ms"], summary["model_latency_ms"]
    print(
        f"total={summary['total_requests']} ok={summary['successful']} erros={summary['errors']} "
        f"({summary['error_rate']:.2%}) rps={summary['achieved_rps']:.1f}"
    )
    print(
        f"latencia HTTP  (ms): p50={http['p50']:.2f} p95={http['p95']:.2f} p99={http['p99']:.2f} max={http['max']:.2f}"
    )
    print(f"latencia modelo(ms): p50={model['p50']:.3f} p95={model['p95']:.3f} p99={model['p99']:.3f}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"resumo salvo em {args.output}")
    return 1 if summary["error_rate"] > 0.05 else 0


if __name__ == "__main__":
    sys.exit(main())
