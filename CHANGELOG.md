# Changelog

Formato baseado em [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/); versionamento [SemVer](https://semver.org/lang/pt-BR/).

## [1.1.0] - 2026-09-12

### Adicionado
- `POST /model/reload`: recarrega o modelo promovido sem reiniciar a API; em falha, o modelo atual continua servindo. Métrica `triage_model_reloads_total`.
- Autenticação por API key (`TRIAGE_API_KEY`, header `X-API-Key`) nas rotas de inferência e no reload; rotas operacionais permanecem abertas.
- Limite de taxa por cliente (`TRIAGE_RATE_LIMIT_PER_MINUTE`, janela deslizante, `429` + `Retry-After`). Métrica `triage_rate_limited_total`.
- Múltiplos workers uvicorn (`TRIAGE_WORKERS`) com métricas Prometheus em modo multiprocesso (`docker/entrypoint.sh`).
- Relatório de calibração (Brier, ECE, MCE, diagrama de confiabilidade) no estágio de treino, nos metadados do modelo e no MLflow.
- Experimento de generalização em corpus público real (`scripts/experiment_public_dataset.py`, Medical Abstracts TC Corpus).
- DAG: task `reload_api`, parâmetros de trigger (`n_samples`, `seed`, gates), `execution_timeout`, callbacks que gravam `reports/last_run.json`; o quality gate agora falha o run em vez de pular.
- CI: smoke test com API key e reload, teste com 3 workers, tamanho da imagem, varredura Trivy, `--cov-fail-under=85`, experimento público no job de pipeline.
- Dependabot (GitHub Actions, pip, Docker), actions fixadas por SHA, `permissions: contents: read`.
- Testes para segurança, reload, calibração, formatadores de log, logging no MLflow e script de carga (91% de cobertura).
- `CONTRIBUTING.md` e este changelog.

### Alterado
- `evaluate_pipeline` aceita qualquer conjunto de classes (`label_list`), permitindo reutilizar o pipeline em outros datasets.
- `triage_model_info` é limpo a cada (re)carga para não deixar séries órfãs.
- Script de carga usa `127.0.0.1` por padrão (no Windows, `localhost` resolve para `::1` primeiro).

## [1.0.0] - 2026-09-10

### Adicionado
- Gerador determinístico de laudos sintéticos em português e validação de contrato do dataset.
- Pipeline de treino com três candidatos (Regressão Logística, Complement NB, Random Forest), seleção por F1 macro e metadados com SHA-256.
- Exportação para ONNX com verificação de paridade; benchmark de latência scikit-learn × ONNX Runtime.
- API FastAPI (`/predict`, `/predict/batch`, `/health`, `/ready`, `/model/info`, `/metrics`) com backend selecionável (ONNX ou scikit-learn).
- Instrumentação Prometheus, middleware com `X-Request-ID`, logs JSON.
- Docker multi-stage não-root, Compose com Prometheus + Grafana (dashboard provisionado, alertas), profiles para carga, MLflow e Airflow.
- DAG Airflow de retreino com quality gate e promoção; imagem própria do Airflow.
- CI (lint, testes em matriz, pipeline smoke, integridade da DAG, build + smoke test Docker) e release para o GHCR.
- Documentação: README, decisão de arquitetura em nuvem, monitoramento, latência, Model Card, ADRs.
