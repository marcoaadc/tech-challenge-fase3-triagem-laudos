"""Autenticacao por API key e limite de taxa por cliente.

Ambos sao opcionais e desligados por padrao (``TRIAGE_API_KEY`` vazio, ``TRIAGE_RATE_LIMIT_PER_MINUTE=0``),
para que o Compose local e o CI funcionem sem segredos. Em producao, defina os dois.

O limite de taxa e uma janela deslizante por IP mantida em memoria: suficiente para um processo;
com varias replicas, substitua por um backend compartilhado (Redis) ou pelo rate limit do gateway/ALB.
"""

from __future__ import annotations

import hmac
import threading
import time
from collections import deque

from fastapi import HTTPException, Request, status

from triage.api.metrics import RATE_LIMITED_TOTAL
from triage.config.settings import Settings

API_KEY_HEADER = "X-API-Key"


class SlidingWindowRateLimiter:
    """Permite ate ``limit`` eventos por ``window_seconds`` por chave (thread-safe)."""

    def __init__(self, limit: int, window_seconds: float = 60.0) -> None:
        self.limit = limit
        self.window = window_seconds
        self._events: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, now: float | None = None) -> tuple[bool, float]:
        """Retorna (permitido, segundos ate liberar)."""
        now = time.monotonic() if now is None else now
        with self._lock:
            events = self._events.setdefault(key, deque())
            cutoff = now - self.window
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= self.limit:
                return False, max(events[0] + self.window - now, 0.0)
            events.append(now)
            return True, 0.0

    def reset(self) -> None:
        with self._lock:
            self._events.clear()


def client_key(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _settings(request: Request) -> Settings:
    return request.app.state.settings


async def require_api_key(request: Request) -> None:
    """Dependencia: exige ``X-API-Key`` quando ``TRIAGE_API_KEY`` esta configurada."""
    settings = _settings(request)
    if not settings.api_key_enabled:
        return
    provided = request.headers.get(API_KEY_HEADER, "")
    if not provided or not hmac.compare_digest(provided, settings.api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key ausente ou invalida",
            headers={"WWW-Authenticate": "ApiKey"},
        )


async def enforce_rate_limit(request: Request) -> None:
    """Dependencia: aplica o limite por minuto quando configurado."""
    limiter: SlidingWindowRateLimiter | None = getattr(request.app.state, "rate_limiter", None)
    if limiter is None:
        return
    route = getattr(request.scope.get("route"), "path", request.url.path)
    allowed, retry_after = limiter.allow(client_key(request))
    if not allowed:
        RATE_LIMITED_TOTAL.labels(path=route).inc()
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"limite de {limiter.limit} requisicoes por minuto excedido",
            headers={"Retry-After": str(max(int(retry_after) + 1, 1))},
        )
