"""Rotas da API de triagem."""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Request, status

from triage import __version__
from triage.api.metrics import MODEL_RELOADS_TOTAL, observe_prediction
from triage.api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    ModelInfoResponse,
    ModelSummary,
    PredictRequest,
    PredictResponse,
    ReadinessResponse,
    ReloadResponse,
)
from triage.api.security import enforce_rate_limit, require_api_key
from triage.config.settings import Settings
from triage.serving.predictor import ModelNotFoundError, Predictor

logger = logging.getLogger("triage.api")

router = APIRouter()
protected = [Depends(require_api_key), Depends(enforce_rate_limit)]


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


def _summary(predictor: Predictor) -> ModelSummary:
    return ModelSummary(
        versao=predictor.metadata.model_version, tipo=predictor.metadata.model_type, backend=predictor.backend
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
    "/model/reload",
    response_model=ReloadResponse,
    tags=["modelo"],
    summary="Recarrega o modelo a partir do diretorio configurado",
    dependencies=[Depends(require_api_key)],
)
def model_reload(request: Request) -> ReloadResponse:
    """Troca o modelo em memoria pelo que estiver em ``models_dir`` (ex.: apos a DAG promover um novo).

    Se a carga falhar, o modelo atual continua servindo e a resposta e 503.
    """
    from triage.api.main import load_model_into

    previous = getattr(request.app.state, "predictor", None)
    previous_version = previous.metadata.model_version if previous else None
    start = time.perf_counter()
    try:
        predictor = load_model_into(request.app, _get_settings(request))
    except (ModelNotFoundError, ValueError, OSError) as exc:
        MODEL_RELOADS_TOTAL.labels(result="failure").inc()
        logger.error("falha ao recarregar o modelo: %s", exc)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"recarga falhou: {exc}") from exc
    MODEL_RELOADS_TOTAL.labels(result="success").inc()
    return ReloadResponse(
        status="reloaded",
        versao_anterior=previous_version,
        versao_atual=predictor.metadata.model_version,
        alterado=previous_version != predictor.metadata.model_version,
        duracao_ms=round((time.perf_counter() - start) * 1000, 3),
        modelo=_summary(predictor),
    )


@router.post(
    "/predict",
    response_model=PredictResponse,
    tags=["inferencia"],
    summary="Classifica a urgencia de um laudo",
    dependencies=protected,
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
        modelo=_summary(predictor),
    )


@router.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
    tags=["inferencia"],
    summary="Classifica um lote de laudos",
    dependencies=protected,
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
    summary = _summary(predictor)
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
