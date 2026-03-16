from __future__ import annotations

import time
import threading
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response


_BUCKET_TTL = 300.0  # seconds before stale bucket entries are pruned
_CLEANUP_INTERVAL = 60.0  # seconds between cleanup sweeps


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Token bucket rate limiter per client IP."""

    def __init__(self, app, rpm: int = 120) -> None:  # type: ignore[no-untyped-def]
        super().__init__(app)
        self.rpm = rpm
        self.tokens_per_second = rpm / 60.0
        self._buckets: dict[str, tuple[float, float]] = {}  # ip -> (tokens, last_refill)
        self._lock = threading.Lock()
        self._last_cleanup = time.monotonic()

    def _get_client_ip(self, request: Request) -> str:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        client = request.client
        return client.host if client else "unknown"

    def _cleanup_stale(self, now: float) -> None:
        if now - self._last_cleanup < _CLEANUP_INTERVAL:
            return
        self._last_cleanup = now
        stale = [ip for ip, (_, ts) in self._buckets.items() if now - ts > _BUCKET_TTL]
        for ip in stale:
            del self._buckets[ip]

    def _consume(self, ip: str) -> tuple[bool, float]:
        now = time.monotonic()
        with self._lock:
            self._cleanup_stale(now)
            if ip in self._buckets:
                tokens, last_refill = self._buckets[ip]
                elapsed = now - last_refill
                tokens = min(float(self.rpm), tokens + elapsed * self.tokens_per_second)
            else:
                tokens = float(self.rpm)
                last_refill = now

            if tokens >= 1.0:
                self._buckets[ip] = (tokens - 1.0, now)
                return True, tokens - 1.0
            self._buckets[ip] = (tokens, now)
            return False, tokens

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        ip = self._get_client_ip(request)
        allowed, remaining = self._consume(ip)
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={"detail": "rate limit exceeded"},
                headers={
                    "RateLimit-Limit": str(self.rpm),
                    "RateLimit-Remaining": "0",
                    "Retry-After": "1",
                },
            )
        response = await call_next(request)
        response.headers["RateLimit-Limit"] = str(self.rpm)
        response.headers["RateLimit-Remaining"] = str(int(remaining))
        return response
