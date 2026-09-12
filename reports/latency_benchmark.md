# Benchmark de latencia - modelo 20260912.125820-440596bb (tfidf+logreg)

Ambiente: Python 3.10.4 | Windows-10-10.0.19044-SP0 | Intel64 Family 6 Model 140 Stepping 1, GenuineIntel

Tamanho dos artefatos: scikit-learn 146 KB | ONNX 225 KB

| Backend | Modo | Batch | Chamadas | p50 (ms) | p90 (ms) | p95 (ms) | p99 (ms) | Media (ms) | Throughput (laudos/s) |
|---|---|---|---|---|---|---|---|---|---|
| sklearn | single | 1 | 2000 | 0.550 | 0.658 | 0.830 | 1.223 | 0.575 | 1734 |
| sklearn | batch | 32 | 62 | 3.494 | 4.242 | 4.756 | 6.540 | 3.587 | 8895 |
| onnx | single | 1 | 2000 | 0.141 | 0.220 | 0.265 | 0.385 | 0.159 | 6258 |
| onnx | batch | 32 | 62 | 3.460 | 4.600 | 5.233 | 6.113 | 3.666 | 8700 |

**Speedup ONNX vs scikit-learn (inferencia unitaria):** 3.9x no p50, 3.1x no p95.
