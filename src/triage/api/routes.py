"""Rotas da API de triagem."""

from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException, Request, status

from triage import __version__
from triage.api.metrics import observe_prediction
from triage.api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    ModelInfoResponse,
    ModelSummary,
    PredictRequest,
    PredictResponse,
    ReadinessResponse,
)
from triage.config.settings import Settings
from triage.serving.predictor import Predictor

router = APIRouter()


def _get_predictor(request: Request) -> Predictor:
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="modelo nao carregado")
    return predictor


def _get_settings(request: Request) -> Settings:
    return request.app.state.settings


def _validate_length(text: str, settings: Settings) -> None:
    if len(text) > settings.max_text_length:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"texto excede o limite de {settings.max_text_length} caracteres",
        )


@router.get("/health", response_model=HealthResponse, tags=["operacional"], summary="Liveness probe")
def health(request: Request) -> HealthResponse:
    settings = _get_settings(request)
    return HealthResponse(status="ok", app=settings.app_name, versao_app=__version__, ambiente=settings.environment)


@router.get("/ready", response_model=ReadinessResponse, tags=["operacional"], summary="Readiness probe")
def ready(request: Request) -> ReadinessResponse:
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="modelo nao carregado")
    return ReadinessResponse(
        status="ready",
        modelo_carregado=True,
        backend=predictor.backend,
        versao_modelo=predictor.metadata.model_version,
    )


@router.get("/model/info", response_model=ModelInfoResponse, tags=["modelo"], summary="Metadados do modelo carregado")
def model_info(request: Request) -> ModelInfoResponse:
    predictor = _get_predictor(request)
    meta = predictor.metadata
    return ModelInfoResponse(
        versao=meta.model_version,
        tipo=meta.model_type,
        backend=predictor.backend,
        classes=predictor.classes,
        treinado_em=meta.trained_at,
        metricas=meta.metrics,
        hiperparametros=meta.hyperparameters,
        artefatos=meta.artifacts,
    )


@router.post(
    "/predict", response_model=PredictResponse, tags=["inferencia"], summary="Classifica a urgencia de um laudo"
)
def predict(payload: PredictRequest, request: Request) -> PredictResponse:
    predictor = _get_predictor(request)
    settings = _get_settings(request)
    _validate_length(payload.texto, settings)

    start = time.perf_counter()
    (prediction,) = predictor.predict([payload.texto])
    elapsed = time.perf_counter() - start
    observe_prediction(predictor.backend, prediction.label, prediction.confidence, elapsed)

    return PredictResponse(
        classe=prediction.label,
        confianca=prediction.confidence,
        probabilidades=prediction.probabilities,
        latencia_ms=round(elapsed * 1000, 3),
        modelo=ModelSummary(
            versao=predictor.metadata.model_version, tipo=predictor.metadata.model_type, backend=predictor.backend
        ),
    )


@router.post(
    "/predict/batch", response_model=BatchPredictResponse, tags=["inferencia"], summary="Classifica um lote de laudos"
)
def predict_batch(payload: BatchPredictRequest, request: Request) -> BatchPredictResponse:
    predictor = _get_predictor(request)
    settings = _get_settings(request)
    if len(payload.laudos) > settings.max_batch_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"lote excede o maximo de {settings.max_batch_size} laudos",
        )
    for text in payload.laudos:
        if not text.strip():
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="laudo vazio no lote")
        _validate_length(text, settings)

    start = time.perf_counter()
    predictions = predictor.predict(payload.laudos)
    elapsed = time.perf_counter() - start
    per_item = elapsed / max(len(predictions), 1)
    summary = ModelSummary(
        versao=predictor.metadata.model_version, tipo=predictor.metadata.model_type, backend=predictor.backend
    )
    results = []
    for p in predictions:
        observe_prediction(predictor.backend, p.label, p.confidence, per_item)
        results.append(
            PredictResponse(
                classe=p.label,
                confianca=p.confidence,
                probabilidades=p.probabilities,
                latencia_ms=round(per_item * 1000, 3),
                modelo=summary,
            )
        )
    return BatchPredictResponse(resultados=results, total=len(results), latencia_ms=round(elapsed * 1000, 3))
