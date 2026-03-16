from __future__ import annotations

import logging
import traceback

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("api.error")


def _get_correlation_id() -> str:
    try:
        from packages.logging_config.setup import correlation_id_var
        return correlation_id_var.get("")
    except (ImportError, LookupError):
        return ""


def register_error_handlers(app: FastAPI) -> None:
    """Register global error handlers for the application."""

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content={
                "detail": str(exc),
                "correlation_id": _get_correlation_id(),
                "status_code": 400,
            },
        )

    @app.exception_handler(KeyError)
    async def key_error_handler(request: Request, exc: KeyError) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content={
                "detail": f"not found: {exc}",
                "correlation_id": _get_correlation_id(),
                "status_code": 404,
            },
        )

    @app.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
        if isinstance(exc, HTTPException):
            raise exc
        logger.error(
            "unhandled exception: %s",
            exc,
            extra={"traceback": traceback.format_exc()},
        )
        return JSONResponse(
            status_code=500,
            content={
                "detail": "internal server error",
                "correlation_id": _get_correlation_id(),
                "status_code": 500,
            },
        )
