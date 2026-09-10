"""Contratos de entrada/saida da API (Pydantic v2)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from triage.data.generator import LABELS

EXAMPLE_LAUDO = (
    "RADIOGRAFIA DE TÓRAX EM PA E PERFIL. Paciente masculino, 67 anos. "
    "Pneumotórax volumoso à direita com desvio contralateral do mediastino. "
    "ACHADO CRÍTICO - comunicado ao médico solicitante."
)


class PredictRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{"texto": EXAMPLE_LAUDO}]})

    texto: str = Field(..., min_length=1, description="Texto livre do laudo ou nota clínica.")


class BatchPredictRequest(BaseModel):
    laudos: list[str] = Field(..., min_length=1, description="Lista de textos de laudos.")


class ModelSummary(BaseModel):
    versao: str
    tipo: str
    backend: str


class PredictResponse(BaseModel):
    classe: str = Field(..., description=f"Classe de urgência prevista: {', '.join(LABELS)}.")
    confianca: float = Field(..., ge=0.0, le=1.0, description="Probabilidade da classe prevista.")
    probabilidades: dict[str, float] = Field(..., description="Probabilidade por classe.")
    latencia_ms: float = Field(..., description="Tempo de inferência do modelo (ms), sem overhead HTTP.")
    modelo: ModelSummary


class BatchPredictResponse(BaseModel):
    resultados: list[PredictResponse]
    total: int
    latencia_ms: float = Field(..., description="Tempo total de inferência do lote (ms).")


class HealthResponse(BaseModel):
    status: str
    app: str
    versao_app: str
    ambiente: str


class ReadinessResponse(BaseModel):
    status: str
    modelo_carregado: bool
    backend: str | None = None
    versao_modelo: str | None = None


class ModelInfoResponse(BaseModel):
    versao: str
    tipo: str
    backend: str
    classes: list[str]
    treinado_em: str
    metricas: dict
    hiperparametros: dict
    artefatos: dict
