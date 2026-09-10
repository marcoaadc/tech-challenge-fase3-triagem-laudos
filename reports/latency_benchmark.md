# Benchmark de latencia - modelo 20260910.213610-440596bb (tfidf+logreg)

Ambiente: Python 3.10.4 | Windows-10-10.0.19044-SP0 | Intel64 Family 6 Model 140 Stepping 1, GenuineIntel

Tamanho dos artefatos: scikit-learn 146 KB | ONNX 225 KB

| Backend | Modo | Batch | Chamadas | p50 (ms) | p90 (ms) | p95 (ms) | p99 (ms) | Media (ms) | Throughput (laudos/s) |
|---|---|---|---|---|---|---|---|---|---|
| sklearn | single | 1 | 2000 | 0.554 | 0.769 | 0.910 | 1.388 | 0.598 | 1667 |
| sklearn | batch | 32 | 62 | 3.470 | 4.513 | 4.752 | 5.397 | 3.579 | 8913 |
| onnx | single | 1 | 2000 | 0.129 | 0.170 | 0.202 | 0.305 | 0.137 | 7239 |
| onnx | batch | 32 | 62 | 3.250 | 3.937 | 4.110 | 4.487 | 3.348 | 9523 |

**Speedup ONNX vs scikit-learn (inferencia unitaria):** 4.3x no p50, 4.5x no p95.
