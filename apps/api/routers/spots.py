from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from apps.api.routers.utils import get_container
from packages.auth.dependencies import get_current_user
from packages.auth.models import User

router = APIRouter(prefix="/v1/spots")


@router.get("/packs")
def list_spot_packs(request: Request) -> dict:
    container = get_container(request)
    return {"packs": container.spot_packs.list_packs()}


@router.get("/taxonomy")
def taxonomy_catalog(request: Request) -> dict:
    container = get_container(request)
    return container.taxonomy.catalog()


@router.post("/packs/{spot_id}/instantiate")
def instantiate_spot_pack(spot_id: str, request: Request, user: User = Depends(get_current_user)) -> dict:
    container = get_container(request)
    try:
        runtime, pack = container.spot_packs.instantiate(spot_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="spot pack not found") from exc
    container.runtimes[runtime.state.hand_id] = runtime
    snapshot = container.engine.state_snapshot(runtime)
    return {"pack": pack, "snapshot": snapshot}


@router.post("/packs/{spot_id}/compare-solver-like")
def compare_solver_like(spot_id: str, request: Request, user: User = Depends(get_current_user)) -> dict:
    container = get_container(request)
    try:
        return container.solver_labels.compare_spot_pack(spot_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="spot pack not found") from exc
