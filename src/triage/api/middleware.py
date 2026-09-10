"""Middleware ASGI: correlacao de requisicoes, metricas HTTP e log de acesso."""

from __future__ import annotations

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from triage.api.metrics import EXCEPTIONS_TOTAL, HTTP_IN_PROGRESS, HTTP_REQUEST_DURATION, HTTP_REQUESTS_TOTAL

logger = logging.getLogger("triage.access")

REQUEST_ID_HEADER = "X-Request-ID"
PROCESS_TIME_HEADER = "X-Process-Time-Ms"


def _route_template(request: Request) -> str:
    """Usa o template da rota (``/predict``) e nao o path bruto, limitando a cardinalidade das metricas."""
    route = request.scope.get("route")
    path = getattr(route, "path", None)
    if path:
        return path
    return "unmatched"


class ObservabilityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
        request.state.request_id = request_id
        method = request.method
        start = time.perf_counter()
        HTTP_IN_PROGRESS.inc()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as exc:  # pragma: no cover - repassado ao handler global do FastAPI
            EXCEPTIONS_TOTAL.labels(type=type(exc).__name__).inc()
            raise
        finally:
            elapsed = time.perf_counter() - start
            path = _route_template(request)
            # /metrics nao e contabilizado para nao poluir as series com o proprio scraping.
            if path != "/metrics":
                HTTP_REQUESTS_TOTAL.labels(method=method, path=path, status=str(status_code)).inc()
                HTTP_REQUEST_DURATION.labels(method=method, path=path).observe(elapsed)
                logger.info(
                    "request",
                    extra={
                        "request_id": request_id,
                        "method": method,
                        "path": request.url.path,
                        "status": status_code,
                        "duration_ms": round(elapsed * 1000, 3),
                    },
                )
            HTTP_IN_PROGRESS.dec()

        response.headers[REQUEST_ID_HEADER] = request_id
        response.headers[PROCESS_TIME_HEADER] = f"{elapsed * 1000:.3f}"
        return response
