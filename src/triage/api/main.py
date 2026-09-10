"""Aplicacao FastAPI: fabrica da app, ciclo de vida do modelo e tratamento de erros."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from triage import __version__
from triage.api.logging_config import configure_logging
from triage.api.metrics import EXCEPTIONS_TOTAL, MODEL_INFO, REGISTRY
from triage.api.middleware import ObservabilityMiddleware
from triage.api.routes import router
from triage.config.settings import Settings, get_settings
from triage.serving.predictor import ModelNotFoundError, load_predictor

logger = logging.getLogger("triage.api")

DESCRIPTION = """
API de **triagem automática de laudos médicos**: recebe o texto livre de um laudo ou nota clínica e
retorna a classe de urgência (`normal`, `atencao`, `urgente`) com as probabilidades por classe.

- Modelo: TF-IDF (1-2 gramas) + classificador linear, exportado para **ONNX Runtime**.
- Observabilidade: métricas Prometheus em `/metrics`, logs estruturados e `X-Request-ID`.
"""


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_json)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings
        app.state.predictor = None
        try:
            predictor = load_predictor(settings)
            warmup_ms = predictor.warmup()
            app.state.predictor = predictor
            MODEL_INFO.labels(
                version=predictor.metadata.model_version,
                type=predictor.metadata.model_type,
                backend=predictor.backend,
            ).set(1)
            logger.info(
                "modelo carregado",
                extra={
                    "backend": predictor.backend,
                    "model_version": predictor.metadata.model_version,
                    "warmup_ms": round(warmup_ms, 3),
                },
            )
        except ModelNotFoundError as exc:
            # A API sobe (liveness ok) mas /ready e /predict respondem 503 ate existir modelo.
            logger.error("modelo indisponivel: %s", exc)
        yield
        app.state.predictor = None

    app = FastAPI(
        title="Triagem de Laudos Medicos",
        version=__version__,
        description=DESCRIPTION,
        lifespan=lifespan,
        contact={"name": "Tech Challenge FIAP - Fase 3"},
    )
    app.add_middleware(ObservabilityMiddleware)
    app.include_router(router)

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        EXCEPTIONS_TOTAL.labels(type=type(exc).__name__).inc()
        logger.exception("erro nao tratado", extra={"path": request.url.path})
        return JSONResponse(status_code=500, content={"detail": "erro interno"})

    return app


app = create_app()


def run() -> None:
    """Entrypoint de linha de comando (``triage-api``)."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "triage.api.main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":  # pragma: no cover
    run()
