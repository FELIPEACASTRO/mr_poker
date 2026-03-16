from __future__ import annotations

import time
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest

    request_count = Counter(
        "http_requests_total",
        "Total HTTP requests",
        ["method", "path", "status"],
    )
    request_duration = Histogram(
        "http_request_duration_seconds",
        "HTTP request duration",
        ["method", "path"],
    )
    active_hands = Gauge("poker_active_hands", "Number of active hands in memory")
    engine_hands_total = Counter("poker_engine_hands_total", "Total hands processed by engine")
    _HAS_PROMETHEUS = True
except ImportError:
    _HAS_PROMETHEUS = False
    request_count = None  # type: ignore[assignment]
    request_duration = None  # type: ignore[assignment]
    active_hands = None  # type: ignore[assignment]
    engine_hands_total = None  # type: ignore[assignment]


class MetricsCollector:
    """Prometheus metrics collector with graceful degradation."""

    @staticmethod
    def inc_request(method: str, path: str, status: int) -> None:
        if _HAS_PROMETHEUS and request_count is not None:
            request_count.labels(method=method, path=path, status=str(status)).inc()

    @staticmethod
    def observe_duration(method: str, path: str, duration: float) -> None:
        if _HAS_PROMETHEUS and request_duration is not None:
            request_duration.labels(method=method, path=path).observe(duration)

    @staticmethod
    def set_active_hands(count: int) -> None:
        if _HAS_PROMETHEUS and active_hands is not None:
            active_hands.set(count)

    @staticmethod
    def inc_engine_hands() -> None:
        if _HAS_PROMETHEUS and engine_hands_total is not None:
            engine_hands_total.inc()


def get_metrics() -> str:
    """Return Prometheus text format metrics."""
    if _HAS_PROMETHEUS:
        return generate_latest().decode("utf-8")
    return ""


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to collect request metrics."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        start = time.monotonic()
        response = await call_next(request)
        duration = time.monotonic() - start
        path = request.url.path
        MetricsCollector.inc_request(request.method, path, response.status_code)
        MetricsCollector.observe_duration(request.method, path, duration)
        return response
