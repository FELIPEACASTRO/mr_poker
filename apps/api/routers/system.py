from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

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
            detail="UI build nao encontrado. Execute o build frontend para habilitar /ui.",
        )

    if path:
        asset_path = (ui_dist / path).resolve()
        try:
            asset_path.relative_to(ui_dist.resolve())
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="path invalido") from exc
        if asset_path.exists() and asset_path.is_file():
            return FileResponse(asset_path)
    return FileResponse(index_file)


@router.get("/health")
def health(request: Request) -> dict[str, str]:
    container = get_container(request)
    return {
        "status": "ok",
        "version": request.app.version,
        "db_path": container.store.db_path,
    }


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
