from __future__ import annotations

import logging
import time
from uuid import uuid4

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("api.request")

try:
    from packages.logging_config.setup import correlation_id_var
except ImportError:
    import contextvars
    correlation_id_var = contextvars.ContextVar("correlation_id", default="")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every request with correlation ID and duration."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        correlation_id = request.headers.get("X-Correlation-ID", str(uuid4()))
        correlation_id_var.set(correlation_id)

        start = time.monotonic()
        try:
            response = await call_next(request)
        except Exception:
            duration_ms = round((time.monotonic() - start) * 1000, 2)
            logger.error(
                "request failed",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "duration_ms": duration_ms,
                    "correlation_id": correlation_id,
                },
            )
            raise

        duration_ms = round((time.monotonic() - start) * 1000, 2)
        response.headers["X-Correlation-ID"] = correlation_id

        logger.info(
            "request completed",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
                "correlation_id": correlation_id,
            },
        )
        return response
