# Tech Challenge FIAP Fase 03 — Triagem Automática de Laudos Médicos

![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)
![ONNX Runtime](https://img.shields.io/badge/inferência-ONNX%20Runtime-005CED.svg)
![Airflow](https://img.shields.io/badge/orquestração-Airflow-017CEE.svg)
![Prometheus](https://img.shields.io/badge/monitoramento-Prometheus%20%2B%20Grafana-E6522C.svg)
![CI](https://github.com/marcoaadc/tech-challenge-fase3-triagem-laudos/actions/workflows/ci.yml/badge.svg)

Sistema de triagem automática de laudos médicos: um classificador de texto (NLP) leve que recebe o texto livre de um laudo ou nota clínica e devolve a prioridade de atendimento — **`normal`**, **`atencao`** ou **`urgente`** — servido em tempo real por uma API FastAPI em container Docker, com pipeline CI/CD (GitHub Actions), retreino orquestrado (Airflow), monitoramento (Prometheus + Grafana) e otimização de latência (ONNX Runtime).

Repositório: <https://github.com/marcoaadc/tech-challenge-fase3-triagem-laudos>

---

## 1. O problema

Um hospital de referência recebe centenas de laudos em texto livre por dia (radiografia, tomografia, laboratório, ECG, ultrassom, notas de pronto-socorro). Sem triagem, a fila é atendida por ordem de chegada e um achado crítico pode esperar atrás de um exame de rotina. O sistema classifica cada laudo **no momento em que é emitido**, para que a fila seja reordenada por urgência.

O modelo é treinado sobre **6.000 laudos sintéticos em português**, gerados por um módulo determinístico a partir de templates clínicos (6 tipos de exame, ~170 achados com severidade anotada) e ruído realista: abreviações, erros de digitação, variação de caixa, negações ("sem sinais de pneumotórax"), 30% dos laudos sem conclusão e 2% de ruído de rótulo. Não existe corpus público em português com laudos rotulados por urgência (os reais, como MIMIC, exigem credenciamento); o gerador torna a ingestão reprodutível e a fonte pode ser trocada por qualquer CSV com colunas `text` e `label`. Para mostrar que o pipeline não depende do gerador, ele também é executado em um corpus médico público real (§9.3).

| | Valor |
|---|---|
| Laudos | **6.000** (treino 4.199 / validação 901 / teste 900) |
| Classes | `normal` 49,5% · `atencao` 30,8% · `urgente` 19,7% |
| Comprimento médio | 232 caracteres |

> Limitações do dataset sintético e considerações éticas estão no [Model Card](docs/model_card.md).

## 2. Arquitetura da solução

```
 ┌──────────────────────────── Pipeline de treino (Airflow DAG / CLIs) ─────────────────────────────┐
 │                                                                                                  │
 │  ingest ──▶ validate ──▶ train ──▶ export_onnx ──▶ quality_gate ──▶ promote ──▶ benchmark ──▶ reload_api
 │  laudos.csv  contrato    3 candidatos  model.onnx   F1≥0.90          models/     sklearn × onnx  POST /model/reload
 │  (gerador)   do dataset  → melhor por  + paridade   recall urg≥0.90  registry    p50/p95/p99     │
 │                          F1 macro (val)             paridade≥0.99                                │
 │                          + calibração                                                            │
 │                              │                                                                   │
 │                              ▼ MLflow (params, métricas por candidato, calibração)               │
 └──────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                        │ models/{model.onnx, model.joblib, metadata.json}
                                                        ▼
 ┌──────────────────────────── Serving + observabilidade (Docker Compose) ──────────────────────────┐
 │                                                                                                  │
 │   cliente ──HTTP──▶ FastAPI (triage-api)  ──▶ Predictor (Strategy: onnx | sklearn)              │
 │   X-API-Key         /predict /predict/batch     normalize_text → ONNX Runtime → probabilidades   │
 │   rate limit        /health /ready /model/info /model/reload                                     │
 │                     /metrics ◀──scrape 5s── Prometheus ──▶ Grafana (dashboard provisionado)      │
 │                                             regras de alerta (down, 5xx, p95, drift)             │
 └──────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                        ▲
 GitHub Actions: lint → test (3.10/3.11) → pipeline smoke → integridade da DAG → build + smoke test Docker → Trivy
 Release (tag v*): publica a imagem no GHCR
```

**O modelo:** TF-IDF (1-2 gramas, `sublinear_tf`) + Regressão Logística com `class_weight="balanced"`, escolhido entre três candidatos (Regressão Logística, Complement Naive Bayes, Random Forest) por F1 macro na validação. O pipeline inteiro (tokenização, TF-IDF e classificador) é exportado para **ONNX** e servido pelo **ONNX Runtime**, com verificação de paridade contra o scikit-learn a cada exportação.

## 3. Estrutura do repositório

```
techallenger3/
├── airflow/
│   ├── dags/triage_training_dag.py   # DAG de retreino (8 tasks, TaskFlow API, params, callbacks)
│   ├── Dockerfile                    # Imagem do Airflow com o pacote instalado
│   └── requirements.txt
├── data/raw/laudos.csv               # Dataset (gerado por `triage.pipelines.ingest`)
├── docker/entrypoint.sh              # Entrypoint da API: workers + métricas multiprocesso
├── docs/
│   ├── arquitetura_cloud.md          # Decisão de deploy em nuvem (Etapa 1)
│   ├── latencia.md                   # Resultados de latência sklearn × ONNX (Etapa 4)
│   ├── monitoramento.md              # Métricas, painéis e alertas (Etapa 3)
│   ├── model_card.md                 # Model Card (métricas, calibração, limitações)
│   └── adr/                          # Decisões de arquitetura (ADRs 001-004)
├── models/                           # Modelo promovido: model.onnx, model.joblib, metadata.json, registry.json
├── monitoring/
│   ├── prometheus/{prometheus.yml, alerts.yml}
│   └── grafana/{provisioning/, dashboards/triage-api.json}
├── reports/                          # Métricas de treino, paridade ONNX, benchmarks, testes de carga, experimento público
├── scripts/
│   ├── load_test.py                  # Teste de carga HTTP / gerador de tráfego (stdlib)
│   └── experiment_public_dataset.py  # Mesmo pipeline no Medical Abstracts TC Corpus
├── src/triage/
│   ├── api/                          # FastAPI: rotas, schemas, segurança, métricas Prometheus, middleware, logging
│   ├── benchmark/                    # Benchmark de latência in-process
│   ├── config/                       # Settings (pydantic-settings, prefixo TRIAGE_)
│   ├── data/                         # Gerador do dataset + validação de contrato
│   ├── nlp/                          # normalize_text (compartilhado treino/inferência)
│   ├── pipelines/                    # CLIs dos estágios: ingest, train, export, promote, benchmark
│   ├── serving/                      # Predictor (Strategy) + factory: OnnxPredictor / SklearnPredictor
│   └── training/                     # Pipeline sklearn, candidatos, avaliação, calibração, exportação ONNX, metadados
├── tests/                            # 89 testes (pytest) — unidade, API, segurança, reload, pipeline, DAG
├── .github/{workflows/ci.yml, workflows/release.yml, dependabot.yml}
├── Dockerfile                        # Multi-stage, usuário não-root, healthcheck, entrypoint
├── docker-compose.yml                # api + prometheus + grafana (+ profiles: load, mlflow, airflow)
├── Makefile · CHANGELOG.md · CONTRIBUTING.md
└── pyproject.toml / poetry.lock
```

## 4. Requisitos

- Python **3.10** ou **3.11**
- [Poetry](https://python-poetry.org/) ≥ 2.0 (lock commitado)
- Docker + Docker Compose v2 (stack de monitoramento e Airflow)
- `make` (opcional; no Windows use Git Bash)

## 5. Instalação

```bash
git clone https://github.com/marcoaadc/tech-challenge-fase3-triagem-laudos.git
cd tech-challenge-fase3-triagem-laudos

poetry install --with train     # ou: make install  (grupo `train`: scikit-learn, skl2onnx, mlflow, pandas)
cp .env.example .env            # opcional; todos os valores têm defaults
```

## 6. Execução rápida

### 6.1 API local (modelo já promovido em `models/`)

```bash
make api                        # uvicorn com reload em http://localhost:8000  (Swagger em /docs)

curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"texto": "RX de tórax. Pneumotórax hipertensivo à esquerda. ACHADO CRÍTICO - comunicado ao médico."}'
```

```json
{
  "classe": "urgente",
  "confianca": 0.97,
  "probabilidades": {"atencao": 0.02, "normal": 0.01, "urgente": 0.97},
  "latencia_ms": 0.21,
  "modelo": {"versao": "20260912.125820-440596bb", "tipo": "tfidf+logreg", "backend": "onnx"}
}
```

Com `TRIAGE_API_KEY` definida, as rotas de inferência exigem o header `X-API-Key`; com `TRIAGE_RATE_LIMIT_PER_MINUTE`, o excesso recebe `429` com `Retry-After`.

### 6.2 Docker: API + Prometheus + Grafana

```bash
docker compose up --build -d              # ou: make docker-up
docker compose --profile load up -d load  # gera tráfego (5 min a 20 req/s) para popular o dashboard
```

| Serviço | URL |
|---|---|
| API / Swagger | <http://localhost:8000/docs> |
| Prometheus | <http://localhost:9090> |
| Grafana (admin/admin) | <http://localhost:3000> — dashboard *Triagem de Laudos - API* já provisionado |

`TRIAGE_WORKERS=3 docker compose up -d api` sobe três processos; o entrypoint ativa o modo multiprocesso do Prometheus e `/metrics` agrega todos.

### 6.3 Airflow (retreino orquestrado)

```bash
docker compose --profile airflow up --build -d   # ou: make airflow-up
# Airflow em http://localhost:8080 (admin/admin) — DAG `triage_training_pipeline`
# MLflow em http://localhost:5000 (o treino da DAG loga params/métricas nele)
```

A DAG grava o modelo promovido em `./models` (volume compartilhado com a API) e, na última task, chama `POST /model/reload` para que a API passe a servi-lo **sem restart**. Pelo botão *Trigger DAG w/ config* é possível ajustar `n_samples`, `seed` e os limiares do quality gate. Cada run grava um resumo em `reports/last_run.json`.

## 7. Pipeline de treino

Cada estágio é uma função `run_*` em `src/triage/pipelines/`, chamada tanto pela CLI quanto pela DAG (testável sem o Airflow):

```bash
make pipeline     # ou, estágio a estágio:
poetry run python -m triage.pipelines.ingest      # gera data/raw/laudos.csv + relatório de validação
poetry run python -m triage.pipelines.train       # treina 3 candidatos + calibração → models/candidate (+ MLflow opcional)
poetry run python -m triage.pipelines.export      # exporta ONNX + paridade sklearn × onnx
poetry run python -m triage.pipelines.promote     # quality gate → copia para models/ + registry.json
poetry run python -m triage.pipelines.benchmark   # latência sklearn × onnx → reports/latency_benchmark.*
make experiment-public                            # mesmo pipeline no Medical Abstracts TC Corpus (§9.3)
```

| Estágio | O que faz | Saída |
|---|---|---|
| `ingest` | Gera o dataset (determinístico por seed) e valida o contrato (≥ 2.000 linhas, classes presentes) | `data/raw/laudos.csv`, `reports/dataset_validation.json` |
| `train` | Split 70/15/15 estratificado; treina os candidatos; escolhe por F1 macro (val) com desempate por recall de `urgente`; avalia no teste e mede a calibração (Brier, ECE, diagrama de confiabilidade) | `models/candidate/{model.joblib, metadata.json}`, `reports/training_metrics.json` |
| `export` | Converte para ONNX (opset 17, `zipmap=False`); mede paridade em 1.000 amostras | `models/candidate/model.onnx`, `reports/onnx_parity.json` |
| `promote` | **Quality gate**: F1 macro ≥ 0,90, recall `urgente` ≥ 0,90, paridade ≥ 0,99; reprova ⇒ falha e o modelo servido permanece | `models/` + `models/registry.json`, `reports/promotion.json` |
| `benchmark` | Latência unitária e em lote dos dois backends | `reports/latency_benchmark.{json,md}` |

Com `--mlflow-tracking-uri` (ou `TRIAGE_MLFLOW_TRACKING_URI`), o estágio `train` registra um run pai com os hiperparâmetros do TF-IDF e as métricas de teste/calibração, e runs aninhados por candidato (`make mlflow-ui` para a interface).

O `metadata.json` do modelo carrega versão, hiperparâmetros, métricas de validação/teste/calibração/paridade e o **SHA-256 do dataset e de cada artefato**, o que permite rastrear qualquer predição da API até o dado que a originou.

## 8. API

| Endpoint | Método | Proteção | Descrição |
|---|---|---|---|
| `/predict` | POST | API key + rate limit | Classifica um laudo (`{"texto": "..."}`) |
| `/predict/batch` | POST | API key + rate limit | Classifica até 64 laudos (`{"laudos": ["...", "..."]}`) |
| `/model/reload` | POST | API key | Recarrega o modelo de `models_dir` sem restart; em falha mantém o atual (503) |
| `/model/info` | GET | — | Versão, métricas, hiperparâmetros e hashes do modelo servido |
| `/health` | GET | — | Liveness: a aplicação está de pé |
| `/ready` | GET | — | Readiness: modelo carregado (503 caso contrário) |
| `/metrics` | GET | — | Métricas Prometheus |
| `/docs` | GET | — | Swagger UI |

Comportamentos relevantes: validação Pydantic (422), limite de tamanho de texto e de lote (413), autenticação por `X-API-Key` comparada em tempo constante (401), limite de taxa por cliente com janela deslizante (429 + `Retry-After`), `X-Request-ID` propagado, `X-Process-Time-Ms` em toda resposta, logs JSON em container, backend selecionável por `TRIAGE_MODEL_BACKEND=onnx|sklearn` sem alterar código. Autenticação e rate limit são desligados por padrão (Compose local e CI sem segredos) e ativados por variável de ambiente; a justificativa está no [ADR 004](docs/adr/004-seguranca-e-recarga-da-api.md).

## 9. Resultados

### 9.1 Qualidade do modelo (teste, 900 laudos)

| Candidato | F1 macro (val) | Recall `urgente` (val) | Treino |
|---|---|---|---|
| **TF-IDF + Regressão Logística** (promovido) | **0,9609** | 0,9494 | 0,5 s |
| TF-IDF + Complement Naive Bayes | 0,9586 | 0,9551 | 0,5 s |
| TF-IDF + Random Forest | 0,9365 | 0,8989 | 1,8 s |

Modelo promovido no **teste**: acurácia **0,9667**, F1 macro **0,9636**, recall de `urgente` **0,9438** (precisão/recall por classe e matriz de confusão no [Model Card](docs/model_card.md)). Paridade scikit-learn × ONNX em 1.000 amostras: **100%** de concordância de rótulos, desvio máximo de probabilidade 0,0095.

**Calibração** (as probabilidades servem de critério clínico, então precisam ser confiáveis): Brier 0,076, **ECE 0,036** (abaixo do limiar usual de 0,05), confiança média 0,931 vs. acurácia 0,967 — o modelo é levemente *sub*confiante, o que é o lado seguro para triagem. Diagrama de confiabilidade no Model Card.

### 9.2 Latência: modelo original × otimizado (Etapa 4)

Benchmark in-process (`reports/latency_benchmark.md`), 2.000 chamadas unitárias, Windows 10 / Intel i7 11ª geração / 1 thread:

| Backend | p50 | p95 | p99 | Throughput |
|---|---|---|---|---|
| scikit-learn (original) | 0,550 ms | 0,830 ms | 1,223 ms | 1.734 laudos/s |
| **ONNX Runtime (otimizado)** | **0,141 ms** | **0,265 ms** | **0,385 ms** | **6.258 laudos/s** |
| Speedup | **3,9×** | **3,1×** | 3,2× | 3,6× |

A comparação ponta a ponta via HTTP (baseline local da Etapa 1, com e sem concorrência), a discussão sobre por que a quantização INT8 foi avaliada e descartada, e as condições de medição estão em [docs/latencia.md](docs/latencia.md).

### 9.3 Generalização: o mesmo pipeline em um corpus público real

`scripts/experiment_public_dataset.py` roda o pipeline, sem nenhum ajuste, no **Medical Abstracts TC Corpus** (Schopf et al., 2022): 11.550 resumos médicos de treino e 2.888 de teste, em inglês, em 5 categorias de doença. Fonte: `reports/public_dataset_experiment.md`.

| Candidato | Acurácia | F1 macro | Vocabulário | Treino |
|---|---|---|---|---|
| Baseline (classe majoritária) | 0,333 | — | — | — |
| TF-IDF + Regressão Logística | 0,512 | 0,510 | 233.677 | 32 s |
| **TF-IDF + Complement NB** | **0,542** | **0,518** | 233.677 | 7 s |

Paridade ONNX: 99,8% dos rótulos, desvio máximo de probabilidade 0,011. O corpus é deliberadamente difícil (categorias que se sobrepõem, classe "general pathological conditions" como guarda-chuva), e os números ficam bem abaixo dos do dataset sintético — o que era esperado e é dito com clareza: o objetivo do experimento é provar que ingestão, treino, avaliação e exportação funcionam em dados reais de outro idioma e com outro conjunto de classes, não competir com o estado da arte naquele corpus. Random Forest foi excluído do padrão por levar dezenas de minutos com 233 mil features esparsas, mais uma evidência a favor do [ADR 002](docs/adr/002-modelo-leve-tfidf-linear.md).

## 10. CI/CD (GitHub Actions)

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) roda a cada push/PR, com actions fixadas por SHA e `permissions: contents: read`:

| Job | O que faz |
|---|---|
| `lint` | `ruff check` + `ruff format --check` |
| `test` | `pytest` com cobertura em Python 3.10 e 3.11 (matriz), `--cov-fail-under=85`; publica `coverage.xml` |
| `pipeline` | Executa ingest → train → export → promote → benchmark em ambiente limpo, roda o experimento no corpus público e publica os relatórios como artefato |
| `dag-integrity` | Instala o Airflow com as constraints oficiais e valida a DAG (import sem erros, 8 tasks, dependências, params, callbacks) |
| `docker` | Valida o compose, builda a imagem (cache GHA), sobe o container com API key e testa `/ready`, `401` sem chave, `/predict` (exige `"classe":"urgente"`), `/model/reload` e `/metrics`; sobe de novo com 3 workers e confere a agregação multiprocesso das métricas; registra o tamanho da imagem; builda a imagem do Airflow |
| `security` | Varredura Trivy da imagem (CVEs altas e críticas com correção disponível) |

[`.github/workflows/release.yml`](.github/workflows/release.yml): ao criar uma tag `vX.Y.Z`, publica a imagem em `ghcr.io/marcoaadc/tech-challenge-fase3-triagem-laudos` com tags semânticas. [`dependabot.yml`](.github/dependabot.yml) mantém actions, dependências Python e imagens base atualizadas.

Hooks de pre-commit (`.pre-commit-config.yaml`) aplicam ruff, verificação de YAML/JSON e bloqueio de arquivos grandes.

## 11. Monitoramento

A API expõe métricas RED (requisições, erros, duração) e métricas de modelo (latência de inferência por backend, predições por classe, confiança, versão servida, recargas, rejeições por limite de taxa). O Compose sobe Prometheus (scrape a cada 5 s, 5 regras de alerta) e Grafana com o dashboard **Triagem de Laudos - API** provisionado — 14 painéis, incluindo total de requisições, latência p50/p95/p99, taxa de erro, inferência por backend e distribuição de classes (drift de saída). Detalhes, consultas PromQL e o plano de monitoramento do modelo: [docs/monitoramento.md](docs/monitoramento.md).

## 12. Decisão arquitetural de deploy (Etapa 1)

**Inferência em tempo real** via API REST (não batch): a urgência de um laudo é perecível e a triagem precisa acontecer no momento da emissão. **AWS ECS Fargate** atrás de um ALB, com Amazon Managed Prometheus/Grafana para observabilidade, MWAA (ou EventBridge + ECS RunTask) para o retreino e S3/ECR para artefatos — o mesmo container e o mesmo dashboard deste repositório, sem gerenciar servidores. Lambda foi descartado pelo cold start; SageMaker Endpoint pelo custo mínimo para um modelo de < 1 MB. Análise completa, diagrama, mapeamento componente → serviço, requisitos não funcionais e estimativa de custo: [docs/arquitetura_cloud.md](docs/arquitetura_cloud.md).

## 13. Decisões de design

| Decisão | Motivação |
|---|---|
| **Strategy + Factory no serving** (`src/triage/serving/predictor.py`) | `OnnxPredictor` e `SklearnPredictor` implementam a mesma interface; `load_predictor(settings)` escolhe pelo `TRIAGE_MODEL_BACKEND`. Permite comparar backends ao vivo e ter fallback sem tocar na API. |
| **Normalização fora do pipeline sklearn** (`src/triage/nlp/preprocessing.py`) | Transformadores customizados não são conversíveis para ONNX; uma única função pura é aplicada no treino e na inferência. |
| **`token_pattern` explícito** (`[a-z0-9][a-z0-9]+`) | O tokenizador do ONNX Runtime interpreta `\b` de forma diferente do regex do Python; o padrão explícito levou a paridade de 99,7% para 100%. |
| **Candidato → quality gate → promoção → reload** (`src/triage/pipelines/promote.py`, `/model/reload`) | O treino nunca escreve direto no diretório servido; um retreino pior não chega à API. A recarga só troca o modelo em memória depois de carregar e aquecer o novo; em falha, o atual permanece. Histórico em `registry.json`. |
| **Segurança desligada por padrão, ligada por ambiente** (`src/triage/api/security.py`) | API key (`hmac.compare_digest`) e rate limit por cliente sem tocar no fluxo local/CI; rotas operacionais ficam abertas para orquestrador e Prometheus. [ADR 004](docs/adr/004-seguranca-e-recarga-da-api.md). |
| **Calibração medida, não forçada** (`src/triage/training/calibration.py`) | Brier/ECE/diagrama de confiabilidade entram nos metadados e no MLflow a cada treino; a Regressão Logística já sai bem calibrada (ECE 0,036), então recalibrar (Platt/isotônica) adicionaria um estágio sem ganho e com risco na exportação ONNX. |
| **Metadados com SHA-256** (`src/triage/training/metadata.py`) | Rastreabilidade: cada modelo referencia o hash do dataset e dos artefatos; `/model/info` expõe tudo. |
| **Métricas por template de rota** (`src/triage/api/middleware.py`) | Evita explosão de cardinalidade no Prometheus; `/metrics` não contabiliza o próprio scrape. |
| **Histograma de inferência separado do HTTP** | Permite distinguir problema de modelo de problema de rede/serialização (painel 10 do dashboard). |
| **Estágios como funções `run_*`** (`src/triage/pipelines/`) | A DAG do Airflow só orquestra; a lógica é testada sem o Airflow e reutilizada pela CLI, pelo CI e pelo experimento no corpus público. |
| **Seeds e determinismo** | Gerador, split e modelos fixam `seed=42`; o mesmo dataset produz o mesmo hash e as mesmas métricas. |
| **pydantic-settings + `.env`** | Configuração tipada com prefixo `TRIAGE_`, defaults locais e override por ambiente (Docker, CI). |
| **Imagem multi-stage, não-root, com healthcheck e entrypoint** (`Dockerfile`, `docker/entrypoint.sh`) | Runtime slim, usuário `appuser`, `HEALTHCHECK` em `/ready`; `TRIAGE_WORKERS` liga o modo multiprocesso do Prometheus automaticamente. |

## 14. Qualidade: testes e lint

```bash
make test        # pytest — 89 testes (preprocessing, gerador, validação, treino, calibração, ONNX, predictor, API, segurança, reload, pipelines, benchmark, logging, MLflow, script de carga, DAG*)
make test-cov    # cobertura (91% em src/triage; o CI exige ≥ 85%)
make lint        # ruff check + ruff format --check
make format
```

\* Os testes da DAG são pulados localmente se o Airflow não estiver instalado e executados no job `dag-integrity` do CI.

## 15. Mapeamento dos requisitos do Tech Challenge

| Requisito | Onde está atendido |
|---|---|
| **Etapa 1** — análise de deploy em nuvem (batch × real-time) no README | §12 + [docs/arquitetura_cloud.md](docs/arquitetura_cloud.md) |
| **Etapa 1** — API FastAPI que recebe o laudo e retorna a classificação | `src/triage/api/` (§8) |
| **Etapa 1** — API em Docker + baseline de latência local | `Dockerfile`, `docker-compose.yml`; [docs/latencia.md](docs/latencia.md) |
| **Etapa 2** — workflow GitHub Actions com lint e testes a cada push | `.github/workflows/ci.yml` (6 jobs) (§10) |
| **Etapa 2** — DAG Airflow (ler CSV → treinar → salvar modelo) | `airflow/dags/triage_training_dag.py` (8 tasks, quality gate, reload da API) (§6.3, §7) |
| **Etapa 3** — instrumentação com `prometheus_client` (tempo de requisição, contagem) | `src/triage/api/metrics.py`, `middleware.py` |
| **Etapa 3** — docker-compose com API + Prometheus + Grafana | `docker-compose.yml`, `monitoring/` |
| **Etapa 3** — dashboard Grafana (≥ 3 painéis) + JSON | `monitoring/grafana/dashboards/triage-api.json` (14 painéis) + [docs/monitoramento.md](docs/monitoramento.md) |
| **Etapa 4** — treinar o classificador de texto | `src/triage/training/`, `triage.pipelines.train` (§9.1, §9.3) |
| **Etapa 4** — técnica de otimização (ONNX) e comparação de latência | `src/triage/training/export.py`, `src/triage/benchmark/`, `reports/latency_benchmark.md`, [docs/latencia.md](docs/latencia.md) |
| Bibliotecas: scikit-learn, FastAPI, prometheus-client, Airflow | `pyproject.toml`, `airflow/requirements.txt` |
| CI/CD com ≥ 2 automações | lint, testes, pipeline, DAG, docker, security (+ release, dependabot) |
| Histórico de commits semântico | Conventional Commits (`feat:`, `fix:`, `ci:`, `docs:`, …) + [CHANGELOG](CHANGELOG.md) |
| Dataset com texto + target e ≥ 2.000 amostras | `data/raw/laudos.csv` (6.000), validado por `triage.data.validation`; corpus público de 14.438 resumos no experimento de generalização |

## 16. Documentação adicional

- [Decisão arquitetural de deploy em nuvem](docs/arquitetura_cloud.md)
- [Latência: original × otimizado](docs/latencia.md)
- [Monitoramento e observabilidade](docs/monitoramento.md)
- [Model Card](docs/model_card.md)
- [ADRs](docs/adr/) — 001 tempo real, 002 modelo leve, 003 ONNX Runtime, 004 segurança e recarga da API
- [CHANGELOG](CHANGELOG.md) · [Como contribuir](CONTRIBUTING.md)

---

Projeto desenvolvido para o **Tech Challenge — FIAP, Fase 03** (Machine Learning Engineering).
