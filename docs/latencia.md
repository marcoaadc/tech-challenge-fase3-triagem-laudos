# Latência — Modelo Original × Modelo Otimizado

> Entregáveis da **Etapa 1** (baseline de latência local da API em Docker/local) e da **Etapa 4** (técnica de otimização aplicada e comparação original × otimizado).

## 1. Técnica aplicada: exportação para ONNX Runtime

O pipeline scikit-learn completo (tokenização, TF-IDF 1-2 gramas e Regressão Logística) foi convertido com `skl2onnx` para um único grafo ONNX (opset 17) e é servido pelo **ONNX Runtime** (`CPUExecutionProvider`, otimizações de grafo `ORT_ENABLE_ALL`, `intra_op_num_threads=1`). Toda a inferência — inclusive a vetorização, que no scikit-learn roda em Python puro — passa a executar em C++ em uma chamada.

A exportação é verificada a cada execução do pipeline (`triage.pipelines.export`): em 1.000 laudos, **100% de concordância de rótulos** e desvio máximo de probabilidade de **0,0095** (média 0,0004). O quality gate de promoção bloqueia qualquer exportação com concordância < 0,99.

### Quantização INT8: avaliada e descartada

`onnxruntime.quantization.quantize_dynamic` foi aplicado ao grafo exportado. Resultado: **grafo idêntico**, porque o classificador é emitido como o operador `LinearClassifier` do domínio `ai.onnx.ml`, que não é coberto pelo quantizador (ele atua sobre `MatMul`/`Gemm`). Forçando a exportação sem esse operador (`black_op={"LinearClassifier"}`), o grafo com `MatMul` + `Softmax` ficou **mais lento** (0,118 ms vs 0,081 ms por laudo no experimento) e a quantização falhou na inferência de tipos. Para um modelo linear com ~5 mil features e 3 classes, os pesos ocupam dezenas de KB; o ganho de latência vem do runtime, não da precisão numérica. O experimento está registrado no [ADR 003](adr/003-otimizacao-onnx-runtime.md).

## 2. Benchmark in-process (modelo isolado)

`poetry run python -m triage.pipelines.benchmark --n-calls 2000` · fonte: `reports/latency_benchmark.md`.

Ambiente: Windows 10, Intel Core i7 11ª geração (1 thread de inferência), Python 3.10, scikit-learn 1.7.2, ONNX Runtime 1.23.2. Mesmos 500 laudos do dataset para os dois backends; 20 chamadas de aquecimento descartadas.

| Backend | Modo | Batch | Chamadas | p50 (ms) | p90 (ms) | p95 (ms) | p99 (ms) | Média (ms) | Throughput (laudos/s) |
|---|---|---|---|---|---|---|---|---|---|
| scikit-learn | unitário | 1 | 2000 | 0,550 | 0,658 | 0,830 | 1,223 | 0,575 | 1.734 |
| **ONNX Runtime** | unitário | 1 | 2000 | **0,141** | **0,220** | **0,265** | **0,385** | **0,159** | **6.258** |
| scikit-learn | lote | 32 | 62 | 3,494 | 4,242 | 4,756 | 6,540 | 3,587 | 8.895 |
| ONNX Runtime | lote | 32 | 62 | 3,460 | 4,600 | 5,233 | 6,113 | 3,666 | 8.700 |

**Speedup na inferência unitária (cenário da API): 3,9× no p50, 3,1× no p95, 3,2× no p99.** Execuções anteriores no mesmo ambiente registraram 4,3× a 4,8× no p50; a variação entre execuções é do próprio host (Windows, laptop), não do modelo.

Leitura: no modo unitário o custo do scikit-learn é dominado pelo overhead Python do `TfidfVectorizer.transform` (regex, construção de matriz esparsa, normalização), que o ONNX elimina. Em lote de 32 esse overhead se dilui e os dois backends convergem (a álgebra é a mesma) — o ganho do ONNX é justamente no caso que importa para uma API em tempo real.

Tamanho dos artefatos: `model.joblib` 146 KB · `model.onnx` 225 KB (o ONNX carrega o vocabulário e os pesos IDF de forma explícita).

## 3. Baseline HTTP ponta a ponta (API local)

`python scripts/load_test.py --url http://127.0.0.1:<porta> --requests 2000 --concurrency {1,8}` contra a API rodando com `uvicorn` (1 worker), alternando `TRIAGE_MODEL_BACKEND`. Fonte: `reports/load_test_{onnx,sklearn}_{c1,c8}.json`. A coluna *modelo* é o `latencia_ms` devolvido pela própria API (tempo dentro do `predict`, sem HTTP).

| Backend | Concorrência | HTTP p50 | HTTP p95 | HTTP p99 | Modelo p50 | Modelo p95 | Modelo p99 |
|---|---|---|---|---|---|---|---|
| scikit-learn | 1 | 15,67 ms | 30,41 ms | 32,87 ms | 1,128 ms | 2,104 ms | 2,719 ms |
| **ONNX Runtime** | 1 | 15,35 ms | 28,29 ms | 31,52 ms | **0,393 ms** | **0,679 ms** | **0,924 ms** |
| scikit-learn | 8 | 18,69 ms | 37,21 ms | 54,15 ms | 1,069 ms | 3,879 ms | 5,655 ms |
| **ONNX Runtime** | 8 | 13,00 ms | 31,73 ms | 40,16 ms | **0,480 ms** | **1,351 ms** | **2,298 ms** |

Observações:

- **Na camada do modelo, dentro da API, o ONNX é 2,9× mais rápido no p50 (0,39 vs 1,13 ms) e 2,5× no p99 sob 8 clientes concorrentes (2,3 vs 5,7 ms).** O tempo é maior que no benchmark isolado porque inclui a contenção de threads do servidor e a instrumentação (histogramas Prometheus).
- **A latência HTTP total no Windows é dominada pelo sistema operacional**, não pelo modelo: o quantum do scheduler do Windows (~15,6 ms) e a ausência de `uvloop` produzem p50 na casa de 13–19 ms mesmo com o modelo respondendo em < 1 ms, e uma execução anterior no mesmo ambiente registrou p50 de 4,2 ms para o ONNX — a variância entre execuções é da ordem do próprio quantum. Em Linux/Docker (ambiente alvo, com `uvloop`/`httptools`), o overhead HTTP local fica tipicamente em 1–3 ms, e o ganho do ONNX aparece integralmente no p95/p99 da API. O job `docker` do CI mede a resposta do container em Linux.
- Sob concorrência 8, o p99 da API com scikit-learn sobe para 54 ms (o overhead Python compete pelo GIL entre os threads do servidor), enquanto com ONNX fica em 40 ms — a otimização também melhora a cauda sob carga.
- Nenhuma requisição falhou (0% de erro) em todas as execuções.

## 4. Como reproduzir

```bash
# 1) modelo isolado
poetry run python -m triage.pipelines.benchmark --n-calls 2000

# 2) HTTP, um backend por vez (use 127.0.0.1: no Windows "localhost" resolve para ::1 primeiro)
TRIAGE_MODEL_BACKEND=onnx    poetry run uvicorn triage.api.main:app --port 8002 &
poetry run python scripts/load_test.py --url http://127.0.0.1:8002 --requests 2000 --concurrency 8 --output reports/load_test_onnx_c8.json
TRIAGE_MODEL_BACKEND=sklearn poetry run uvicorn triage.api.main:app --port 8003 &
poetry run python scripts/load_test.py --url http://127.0.0.1:8003 --requests 2000 --concurrency 8 --output reports/load_test_sklearn_c8.json

# 3) no Docker (Linux), com o dashboard do Grafana mostrando a latência de inferência por backend
docker compose up -d && docker compose --profile load up load
```

## 5. Conclusão

| | Original (scikit-learn) | Otimizado (ONNX Runtime) |
|---|---|---|
| Inferência unitária p50 / p95 (isolada) | 0,550 / 0,830 ms | **0,141 / 0,265 ms** |
| Inferência dentro da API p50 / p99 (8 clientes) | 1,069 / 5,655 ms | **0,480 / 2,298 ms** |
| Paridade de rótulos | — | 100% |
| Artefato | 146 KB | 225 KB |
| Dependências no container | scikit-learn + scipy | onnxruntime (scikit-learn mantido apenas como fallback) |

A meta de p95 < 50 ms ponta a ponta é atendida com folga por ambos os backends no ambiente local; o ONNX Runtime é adotado como backend padrão por reduzir o custo do modelo em ~4× no cenário unitário e por melhorar a cauda de latência sob carga, sem qualquer perda de qualidade.
