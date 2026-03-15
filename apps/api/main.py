from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI

from apps.api.container import build_container
from apps.api.routers import (
    benchmark_router,
    governance_router,
    hands_router,
    ml_router,
    sessions_router,
    spots_router,
    system_router,
)


def create_app(db_path: str | None = None, ui_dist_path: str | None = None) -> FastAPI:
    app = FastAPI(title="Poker AI Local API", version="0.30.0")
    container = build_container(db_path=db_path, base_dir=".")
    default_ui_dist = Path(__file__).resolve().parents[1] / "web_ui" / "dist"
    resolved_ui_dist = Path(ui_dist_path).resolve() if ui_dist_path else default_ui_dist.resolve()

    app.include_router(system_router)
    app.include_router(hands_router)
    app.include_router(spots_router)
    app.include_router(sessions_router)
    app.include_router(benchmark_router)
    app.include_router(ml_router)
    app.include_router(governance_router)

    app.state.container = container
    app.state.engine = container.engine
    app.state.store = container.store
    app.state.runtimes = container.runtimes
    app.state.ui_dist_path = str(resolved_ui_dist)
    return app


app = create_app()
