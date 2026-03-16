from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from apps.api.container import build_container
from apps.api.middleware.error_handler import register_error_handlers
from apps.api.middleware.rate_limit import RateLimitMiddleware
from apps.api.middleware.request_logging import RequestLoggingMiddleware
from apps.api.middleware.security_headers import SecurityHeadersMiddleware
from apps.api.routers import (
    benchmark_router,
    governance_router,
    hands_router,
    ml_router,
    sessions_router,
    spots_router,
    system_router,
)
from apps.api.routers.tasks import router as tasks_router
from apps.api.routers.ws import router as ws_router
from packages.config import Settings
from packages.logging_config import setup_logging
from packages.metrics.collector import MetricsMiddleware
from packages.task_queue import TaskQueue
from packages.tracing import setup_tracing

logger = logging.getLogger(__name__)


def create_app(db_path: str | None = None, ui_dist_path: str | None = None) -> FastAPI:
    settings = Settings.from_env()
    setup_logging(settings.log_level)

    app = FastAPI(
        title="Poker AI Local API",
        version="0.30.0",
        description="Local-first Texas Hold'em NL Heads-Up AI training and evaluation platform",
    )

    # Middleware (order matters: last added = first executed)
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(MetricsMiddleware)
    app.add_middleware(RateLimitMiddleware, rpm=settings.rate_limit_rpm)
    app.add_middleware(SecurityHeadersMiddleware)
    if "*" in settings.cors_origins:
        logger.warning(
            "CORS configured with wildcard origin '*' — restrict in production "
            "via POKER_CORS_ORIGINS env var"
        )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-API-Key"],
    )

    # Error handlers
    register_error_handlers(app)

    # Tracing (optional)
    if settings.otel_enabled:
        setup_tracing(service_name="mr_poker")

    # Container + state
    effective_db = db_path or settings.db_path
    container = build_container(db_path=effective_db, base_dir=settings.base_dir)
    resolved_ui_dist = Path(ui_dist_path).resolve() if ui_dist_path else None

    # Routers
    app.include_router(system_router)
    app.include_router(hands_router)
    app.include_router(spots_router)
    app.include_router(sessions_router)
    app.include_router(benchmark_router)
    app.include_router(ml_router)
    app.include_router(governance_router)
    app.include_router(tasks_router)
    app.include_router(ws_router)

    # App state
    app.state.container = container
    app.state.settings = settings
    app.state.engine = container.engine
    app.state.store = container.store
    app.state.runtimes = container.runtimes
    app.state.task_queue = TaskQueue()
    app.state.ui_dist_path = str(resolved_ui_dist)

    logger.info("App created", extra={"env": settings.env, "auth_enabled": settings.auth_enabled})
    return app


app = create_app()
