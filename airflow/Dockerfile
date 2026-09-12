# Imagem do Airflow com o pacote `triage` (e suas dependencias de treino) instalado.
# A DAG chama diretamente as funcoes run_* dos estagios do pipeline.
FROM apache/airflow:3.3.1-python3.10

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
USER airflow

WORKDIR /opt/airflow/project

# Instala as dependencias respeitando as constraints oficiais do Airflow (evita conflitos).
COPY --chown=airflow:root airflow/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.10.5/constraints-3.10.txt"

# Codigo do projeto: instalado em modo editavel para que a DAG importe `triage`.
COPY --chown=airflow:root pyproject.toml README.md ./
COPY --chown=airflow:root src/ ./src/
RUN pip install --no-cache-dir --no-deps -e .

ENV PYTHONPATH=/opt/airflow/project/src
