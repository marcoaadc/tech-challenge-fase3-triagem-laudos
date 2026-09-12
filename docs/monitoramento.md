# Monitoramento e Observabilidade

> Entregável da **Etapa 3**: stack API + Prometheus + Grafana via Docker Compose, dashboard provisionado e regras de alerta.

## 1. Como subir a stack

```bash
docker compose up --build -d          # api (8000) + prometheus (9090) + grafana (3000)
docker compose --profile load up load # opcional: gera tráfego por 5 min a 20 req/s
```

| Serviço | URL | Credenciais |
|---|---|---|
| API (Swagger) | <http://localhost:8000/docs> | — |
| Métricas brutas | <http://localhost:8000/metrics> | — |
| Prometheus | <http://localhost:9090> (targets em `/targets`, alertas em `/alerts`) | — |
| Grafana | <http://localhost:3000> | `admin` / `admin` |

O dashboard **"Triagem de Laudos - API"** é provisionado automaticamente (pasta *Triagem*) a partir de [`monitoring/grafana/dashboards/triage-api.json`](../monitoring/grafana/dashboards/triage-api.json); o datasource Prometheus também é provisionado, sem configuração manual.

Para popular os painéis sem o Docker, com a API rodando localmente:

```bash
python scripts/load_test.py --url http://127.0.0.1:8000 --duration 120 --rps 20
```

## 2. Instrumentação da API

A instrumentação usa `prometheus_client` com um registry dedicado (`triage.api.metrics`) e um middleware ASGI (`triage.api.middleware`) que mede toda requisição pelo **template da rota** (`/predict`, não o path bruto) para manter a cardinalidade das séries sob controle.

| Métrica | Tipo | Labels | O que responde |
|---|---|---|---|
| `triage_http_requests_total` | Counter | `method`, `path`, `status` | Volume e taxa de erro (RED: *Rate*, *Errors*) |
| `triage_http_request_duration_seconds` | Histogram | `method`, `path` | Latência ponta a ponta com percentis (RED: *Duration*) |
| `triage_http_requests_in_progress` | Gauge | — | Saturação instantânea |
| `triage_model_inference_duration_seconds` | Histogram | `backend` | Latência só do modelo, separada do overhead HTTP |
| `triage_predictions_total` | Counter | `label` | Distribuição das classes previstas (drift de saída) |
| `triage_prediction_confidence` | Histogram | — | Confiança das predições (queda = possível drift de entrada) |
| `triage_model_info` | Gauge | `version`, `type`, `backend` | Qual modelo está servindo (rastreabilidade de deploy); zerado e reescrito a cada reload |
| `triage_model_reloads_total` | Counter | `result` | Recargas de modelo em runtime (`success` / `failure`) |
| `triage_rate_limited_total` | Counter | `path` | Requisições rejeitadas por limite de taxa (`429`) |
| `triage_exceptions_total` | Counter | `type` | Exceções não tratadas |

Com mais de um worker uvicorn (`TRIAGE_WORKERS>1`), o entrypoint do container ativa o modo multiprocesso do `prometheus_client` (`PROMETHEUS_MULTIPROC_DIR`) e `/metrics` agrega as séries de todos os processos; o job `docker` do CI valida esse cenário com 3 workers.

Os buckets dos histogramas foram escolhidos para uma API de baixa latência (1 ms a 5 s para HTTP; 50 µs a 500 ms para o modelo), de modo que `histogram_quantile` tenha resolução nos percentis relevantes.

Além das métricas, cada requisição recebe um `X-Request-ID` (propagado se enviado pelo cliente) e um header `X-Process-Time-Ms`, e gera uma linha de log estruturado (JSON em container) com método, rota, status e duração.

## 3. Painéis do dashboard

| # | Painel | Consulta (resumo) | Por que existe |
|---|---|---|---|
| 1 | Total de requisições | `sum(triage_http_requests_total{path=~"/predict.*"})` | Volume acumulado (requisito do desafio) |
| 2 | Requisições/s | `sum(rate(...[1m]))` | Carga atual |
| 3 | Latência p95 `/predict` | `histogram_quantile(0.95, ...)` | SLO de latência (requisito do desafio) |
| 4 | Taxa de erro 5xx | `rate(5xx) / rate(total)` | SLO de erros (requisito do desafio) |
| 5 | Modelo em produção | `triage_model_info` | Versão/backend servidos, visível no deploy |
| 6 | API up | `up{job="triage-api"}` | Disponibilidade do target |
| 7 | Tráfego por rota | `sum by (path, status) (rate(...))` | Mix de rotas e códigos |
| 8 | Latência p50/p95/p99 | `histogram_quantile(...)` | Cauda de latência ao longo do tempo |
| 9 | Taxa de erro 4xx/5xx | razão de rates | Separar erro de cliente de erro do serviço |
| 10 | Latência de inferência do modelo | histograma por `backend` | Isola o custo do modelo (comparação sklearn × ONNX ao vivo) |
| 11 | Predições por classe (janela) | `increase(triage_predictions_total[$__range])` | Distribuição de saída |
| 12 | Distribuição ao longo do tempo | proporção por `label` | Drift de saída / regressão do modelo |
| 13 | Confiança média | `sum/count` do histograma | Queda indica entradas fora da distribuição de treino |
| 14 | Requisições em andamento | gauge | Saturação |

Layout do dashboard (linha superior com os indicadores-chave, séries temporais abaixo):

```
┌────────────┬────────────┬────────────┬────────────┬──────────────────┬──────┐
│ Total req  │ req/s      │ p95 predict│ erro 5xx   │ modelo/backend   │ UP   │
├────────────┴────────────┴────────────┼────────────┴──────────────────┴──────┤
│ Tráfego por rota (req/s)             │ Latência HTTP p50 / p95 / p99        │
├──────────────────┬───────────────────┼──────────────────────────────────────┤
│ Taxa de erro (%) │ Inferência modelo │ Predições por classe (donut)         │
├──────────────────┴───────────────────┼─────────────────────────┬────────────┤
│ Distribuição de classes no tempo     │ Confiança média         │ Em curso   │
└──────────────────────────────────────┴─────────────────────────┴────────────┘
```

## 4. Alertas (Prometheus)

Definidos em [`monitoring/prometheus/alerts.yml`](../monitoring/prometheus/alerts.yml) e visíveis em <http://localhost:9090/alerts>.

| Alerta | Condição | Severidade | Ação sugerida |
|---|---|---|---|
| `TriageApiDown` | `up == 0` por 30 s | critical | Verificar container/health check; clientes devem enfileirar laudos |
| `TriageApiHighErrorRate` | 5xx > 1% por 2 min | critical | Inspecionar logs por `request_id`; considerar rollback da imagem |
| `TriageApiHighLatencyP95` | p95 de `/predict` > 50 ms por 2 min | warning | Verificar CPU/saturação; escalar réplicas |
| `TriageModelInferenceSlow` | p99 do modelo > 5 ms por 5 min | warning | Confirmar backend ONNX, threads e contenção de CPU |
| `TriageUrgentShareAnomaly` | > 50% das predições `urgente` por 15 min | warning | Investigar drift de entrada ou regressão do modelo promovido |

Em nuvem, esses alertas são roteados por Alertmanager (ou pelo equivalente gerenciado) para o canal de plantão do time de dados.

## 5. Plano de monitoramento do modelo (além do dashboard)

| Dimensão | Sinal | Fonte | Frequência |
|---|---|---|---|
| Qualidade offline | F1 macro e recall de `urgente` no teste a cada retreino | `reports/training_metrics.json`, quality gate da DAG | Diária |
| Paridade ONNX | Concordância de rótulos e desvio máximo de probabilidade | `reports/onnx_parity.json` | A cada exportação |
| Latência do modelo | p50/p95 in-process e HTTP | `reports/latency_benchmark.*`, painéis 8 e 10 | A cada promoção / contínuo |
| Drift de saída | Proporção de classes e confiança média | Painéis 12 e 13, alerta `TriageUrgentShareAnomaly` | Contínuo |
| Feedback humano | Discordância entre triagem automática e conduta médica | Sistema clínico (fora do escopo deste repositório) | Semanal |

Quando um sinal de drift dispara, o caminho é: (1) confirmar no Grafana se o comportamento é global ou de um tipo de exame; (2) coletar amostras recentes rotuladas; (3) reexecutar a DAG com o novo dataset; (4) o quality gate decide a promoção.
