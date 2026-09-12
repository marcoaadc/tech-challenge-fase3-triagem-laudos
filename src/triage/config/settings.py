"""Configuracao centralizada da aplicacao (12-factor: tudo via variaveis de ambiente).

Todas as chaves possuem defaults sensatos para execucao local; em container/CI os valores
sao sobrescritos por variaveis de ambiente (prefixo ``TRIAGE_``) ou por um arquivo ``.env``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ModelBackend = Literal["onnx", "sklearn"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TRIAGE_", env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Aplicacao ---------------------------------------------------------
    app_name: str = "triage-api"
    environment: Literal["local", "ci", "docker", "production"] = "local"
    log_level: str = "INFO"
    log_json: bool = Field(default=False, description="Logs em JSON (recomendado em container).")

    # --- Servidor ----------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = Field(default=1, ge=1, description="Processos uvicorn. >1 exige PROMETHEUS_MULTIPROC_DIR.")

    # --- Seguranca ---------------------------------------------------------
    api_key: str | None = Field(
        default=None,
        description="Chave exigida no header X-API-Key em /predict, /predict/batch e /model/reload. Vazio desativa.",
    )
    rate_limit_per_minute: int = Field(
        default=0, ge=0, description="Limite de requisicoes por minuto por cliente nas rotas de inferencia. 0 desativa."
    )

    # --- Modelo ------------------------------------------------------------
    models_dir: Path = Field(default=Path("models"), description="Diretorio com os artefatos do modelo.")
    model_backend: ModelBackend = Field(default="onnx", description="Backend de inferencia usado pela API.")
    onnx_intra_op_threads: int = Field(default=1, ge=0, description="Threads intra-op do ONNX Runtime (0 = auto).")
    max_text_length: int = Field(default=20_000, ge=100, description="Tamanho maximo (chars) aceito em /predict.")
    max_batch_size: int = Field(default=64, ge=1, le=1024)

    # --- Dados / treino ----------------------------------------------------
    data_dir: Path = Path("data")
    reports_dir: Path = Path("reports")
    seed: int = 42
    mlflow_tracking_uri: str | None = Field(default=None, description="Se definido, o treino loga no MLflow.")
    mlflow_experiment: str = "triagem-laudos"

    @property
    def raw_dataset_path(self) -> Path:
        return self.data_dir / "raw" / "laudos.csv"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def api_key_enabled(self) -> bool:
        return bool(self.api_key)


@lru_cache
def get_settings() -> Settings:
    """Instancia unica (cacheada) das configuracoes."""
    return Settings()
