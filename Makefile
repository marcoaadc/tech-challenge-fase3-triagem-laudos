# Tech Challenge FIAP Fase 03 - triagem de laudos medicos
# Requer: Poetry >= 2.0 e make (no Windows, use Git Bash). Docker para os alvos docker-*/airflow-*.

.PHONY: help install lint format test test-cov pipeline api load docker-build docker-up docker-down \
        airflow-up airflow-down mlflow-ui clean

help:
	@echo "Alvos disponiveis:"
	@echo "  install       Instala dependencias (poetry install --with train)"
	@echo "  lint          ruff check + ruff format --check"
	@echo "  format        Formata o codigo e aplica fixes do ruff"
	@echo "  test          Roda a suite de testes"
	@echo "  test-cov      Testes com relatorio de cobertura"
	@echo "  pipeline      ingest -> train -> export -> promote -> benchmark"
	@echo "  experiment-public  Roda o pipeline no Medical Abstracts TC Corpus (generalizacao)"
	@echo "  api           Sobe a API localmente (uvicorn, hot reload)"
	@echo "  load          Teste de carga contra http://127.0.0.1:8000"
	@echo "  docker-build  Builda a imagem da API"
	@echo "  docker-up     Sobe api + prometheus + grafana"
	@echo "  docker-down   Derruba a stack (mantem volumes)"
	@echo "  airflow-up    Sobe Airflow (+ Postgres + MLflow) com a DAG de retreino"
	@echo "  mlflow-ui     UI do MLflow local (file store ./mlruns)"
	@echo "  clean         Remove caches e artefatos temporarios"

install:
	poetry install --with train

lint:
	poetry run ruff check .
	poetry run ruff format --check .

format:
	poetry run ruff format .
	poetry run ruff check --fix .

test:
	poetry run pytest

test-cov:
	poetry run pytest --cov --cov-report=term-missing

pipeline:
	poetry run python -m triage.pipelines.ingest
	poetry run python -m triage.pipelines.train --mlflow-tracking-uri file:./mlruns
	poetry run python -m triage.pipelines.export
	poetry run python -m triage.pipelines.promote
	poetry run python -m triage.pipelines.benchmark

experiment-public:
	poetry run python scripts/experiment_public_dataset.py

api:
	poetry run uvicorn triage.api.main:app --reload --port 8000

load:
	poetry run python scripts/load_test.py --url http://127.0.0.1:8000 --requests 2000 --concurrency 8

docker-build:
	docker build -t triage-api:latest .

docker-up:
	docker compose up --build -d
	@echo "API: http://localhost:8000/docs | Prometheus: http://localhost:9090 | Grafana: http://localhost:3000 (admin/admin)"

docker-down:
	docker compose --profile airflow --profile load --profile mlflow down

airflow-up:
	docker compose --profile airflow up --build -d
	@echo "Airflow: http://localhost:8080 (admin/admin) | MLflow: http://localhost:5000"

airflow-down:
	docker compose --profile airflow down

mlflow-ui:
	poetry run mlflow ui --backend-store-uri file:./mlruns

clean:
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage coverage.xml models/candidate ci-run
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
