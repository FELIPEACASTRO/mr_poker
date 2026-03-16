from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, PlainTextResponse

from apps.api.routers.utils import get_container

router = APIRouter()


def _ui_dist_path(request: Request) -> Path:
    raw_path = getattr(request.app.state, "ui_dist_path", "")
    return Path(raw_path)


def _serve_ui_asset_or_index(request: Request, path: str | None = None) -> FileResponse:
    ui_dist = _ui_dist_path(request)
    index_file = ui_dist / "index.html"
    if not index_file.exists():
        raise HTTPException(
            status_code=503,
            detail="UI build not found. Run frontend build to enable /ui.",
        )

    if path:
        asset_path = (ui_dist / path).resolve()
        try:
            asset_path.relative_to(ui_dist.resolve())
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="invalid path") from exc
        if asset_path.exists() and asset_path.is_file():
            return FileResponse(asset_path)
    return FileResponse(index_file)


@router.get("/health")
def health(request: Request) -> dict:
    container = get_container(request)
    result: dict = {"status": "ok", "version": request.app.version}
    try:
        result["db"] = container.store.db_stats()
    except Exception:
        result["db"] = {"error": "unavailable"}
    return result


@router.get("/health/live")
def health_live(request: Request) -> dict[str, str]:
    return {"status": "ok", "version": request.app.version}


@router.get("/health/ready")
def health_ready(request: Request) -> dict:
    container = get_container(request)
    components = []
    overall = "ok"
    try:
        container.store.get_session("__health_check__")
        components.append({"name": "database", "status": "ok"})
    except Exception as exc:
        components.append({"name": "database", "status": "error", "detail": str(exc)})
        overall = "degraded"
    if overall != "ok":
        raise HTTPException(status_code=503, detail={"status": overall, "components": components})
    return {"status": overall, "components": components}


@router.get("/metrics")
def metrics() -> PlainTextResponse:
    from packages.metrics.collector import get_metrics
    return PlainTextResponse(content=get_metrics(), media_type="text/plain")


@router.get("/")
def root(request: Request) -> dict[str, str]:
    return {
        "service": request.app.title,
        "version": request.app.version,
        "docs": "/docs",
    }


@router.get("/v1/system/readiness")
def system_readiness(request: Request) -> dict:
    container = get_container(request)
    return container.query_bus.get_readiness()


@router.get("/v1/system/alpha-candidate")
def get_alpha_candidate(model_id: str, dataset_id: str, request: Request) -> dict:
    container = get_container(request)
    try:
        return container.query_bus.get_alpha_candidate(model_id, dataset_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="dataset not found") from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="model not found") from exc


@router.get("/v1/deploy/manifest")
def deploy_manifest(request: Request) -> dict:
    container = get_container(request)
    return container.deploy.manifest()


@router.get("/v1/release/notes")
def get_release_notes(request: Request) -> dict:
    container = get_container(request)
    return container.release_notes.current_notes()


@router.get("/ui", include_in_schema=False)
def ui_entrypoint(request: Request) -> FileResponse:
    return _serve_ui_asset_or_index(request, path=None)


@router.get("/ui/{path:path}", include_in_schema=False)
def ui_fallback(request: Request, path: str) -> FileResponse:
    return _serve_ui_asset_or_index(request, path=path)
