"""Estagios do pipeline de treino, executaveis por CLI (``python -m triage.pipelines.<estagio>``).

Cada estagio e uma funcao pura (``run_*``) chamada tanto pela CLI quanto pela DAG do Airflow,
o que permite testar o pipeline sem o Airflow instalado.

    ingest -> train -> export -> promote -> benchmark
"""
