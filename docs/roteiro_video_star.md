# Roteiro do vídeo (5 minutos) — Método STAR

Tech Challenge FIAP Fase 3: deploy do classificador de triagem de laudos com CI/CD, Airflow, Prometheus/Grafana e otimização ONNX.

**Preparação antes de gravar**

- `docker compose up -d` rodando (API, Prometheus, Grafana) e `docker compose --profile load up -d load` gerando tráfego há pelo menos 5 minutos.
- Grafana aberto no dashboard *Triagem de Laudos - API* (janela "Last 15 minutes").
- Airflow aberto (`docker compose --profile airflow up -d`) com a DAG `triage_training_pipeline` já executada uma vez (grafo verde).
- GitHub Actions aberto na última execução do workflow **CI** (jobs lint, test, pipeline, dag-integrity, docker verdes).
- Terminal na raiz do repositório com `reports/latency_benchmark.md` e `reports/load_test_*.json` à mão.
- Editor com abas: `src/triage/api/routes.py`, `src/triage/training/export.py`, `airflow/dags/triage_training_dag.py`, `docker-compose.yml`.

---

## Situation — o problema clínico (0:00 – 0:50)

**Falar**

- Um hospital recebe centenas de laudos em texto livre por dia: radiologia, laboratório, ECG, ultrassom, notas de pronto-socorro.
- A fila é atendida na ordem de chegada; um pneumotórax hipertensivo pode esperar atrás de um exame de rotina.
- Objetivo: triagem automática que classifica cada laudo em **normal / atenção / urgente** no momento em que é assinado, para reordenar a fila.
- Dataset: 6.000 laudos sintéticos em português gerados por templates clínicos (não há corpus público em PT-BR com rótulo de urgência; MIMIC exige credenciamento). Ruído realista: abreviações, erros de digitação, negações, ausência de conclusão, 2% de ruído de rótulo.

**Mostrar**: README §1 e um laudo do `data/raw/laudos.csv`.

## Task — requisitos técnicos (0:50 – 1:30)

**Falar**

- O foco da fase não é o modelo, é o **ciclo de vida em produção**:
  1. API REST em Docker com latência baixa (p95 < 50 ms).
  2. CI/CD no GitHub Actions: lint → testes → pipeline → DAG → build/smoke test da imagem.
  3. Retreino orquestrado no Airflow com quality gate.
  4. Observabilidade: Prometheus + Grafana com painéis de requisições, latência e erro.
  5. Otimização de latência: modelo original vs. otimizado, com números.
- Decisão de arquitetura: inferência em tempo real (não batch) em AWS ECS Fargate (`docs/arquitetura_cloud.md`).

**Mostrar**: tabela de mapeamento de requisitos do README e o diagrama de arquitetura.

## Action — arquitetura e decisões (1:30 – 3:40)

**Falar (ordem sugerida)**

1. **Modelo leve por decisão** (ADR 002): TF-IDF 1-2 gramas + Regressão Logística escolhida entre 3 candidatos por F1 macro na validação; Random Forest ficou para trás (F1 menor e ~40 ms por laudo).
2. **API** (`routes.py`): `/predict`, `/predict/batch`, `/health`, `/ready`, `/model/info`, `/metrics`. Backend de inferência é uma Strategy (`OnnxPredictor` / `SklearnPredictor`) escolhida por variável de ambiente. Lifespan carrega e aquece o modelo; sem modelo, a API sobe mas `/ready` responde 503.
3. **Pipeline de retreino** (`triage_training_dag.py`): `ingest → validate → train → export_onnx → quality_gate → promote → benchmark`. O treino escreve em `models/candidate`; só o gate (F1 ≥ 0,90, recall urgente ≥ 0,90, paridade ONNX ≥ 0,99) copia para `models/` — o modelo em produção nunca é sobrescrito por um pior. Mostrar o grafo no Airflow.
4. **CI/CD** (`ci.yml`): 5 jobs; o job docker sobe o container e faz `curl` no `/predict` exigindo `"classe":"urgente"`. Tag `v*` publica a imagem no GHCR (`release.yml`).
5. **Observabilidade**: middleware mede toda rota pelo template (baixa cardinalidade); histogramas de latência HTTP e de inferência do modelo, contadores por classe e confiança. Compose sobe API + Prometheus + Grafana com dashboard e datasource provisionados; 5 regras de alerta.
6. **Otimização** (`export.py`, ADR 003): exportação para ONNX com verificação de paridade; INT8 avaliado e descartado com justificativa (operador `LinearClassifier` não é quantizável; a variante MatMul ficou mais lenta).

**Mostrar**: grafo da DAG no Airflow; workflow verde no GitHub; dashboard do Grafana com tráfego real; trecho de `export.py`.

## Result — números e lições (3:40 – 5:00)

**Falar**

- Qualidade no teste (900 laudos): acurácia 0,967, F1 macro 0,964, recall de `urgente` 0,94 (ver README §7).
- Latência in-process (2.000 chamadas unitárias): scikit-learn p50 0,55 ms → ONNX p50 0,13 ms (**4,3×**); p95 0,91 ms → 0,20 ms (**4,5×**); 100% de concordância de rótulos.
- Latência HTTP ponta a ponta (3.000 requisições, 8 clientes concorrentes): ver `reports/load_test_onnx.json` vs `reports/load_test_sklearn.json` — o ganho do ONNX aparece também no p95 da API.
- Lições: (1) em modelos lineares, o gargalo é o overhead Python da vetorização, não a álgebra — por isso ONNX ganha e INT8 não; (2) o quality gate na DAG é o que transforma "retreino automático" em algo seguro; (3) medir latência do modelo separada da HTTP no Prometheus evita culpar o modelo por problemas de rede/serialização.
- Próximos passos: dataset real com rótulos de triagem, revisão humana para `atencao` de baixa confiança, deploy no ECS via pipeline.

**Mostrar**: `reports/latency_benchmark.md`, painel de latência de inferência por backend no Grafana, README renderizado no GitHub.

---

**Dicas**: ensaie o tempo por bloco (0:50 / 0:40 / 2:10 / 1:20); deixe os comandos no histórico do terminal; fale do *porquê* das decisões e não leia código linha a linha; grave em 1080p com zoom.
