from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from apps.api.routers.utils import get_container

router = APIRouter(prefix="/v1")


@router.post("/models/train/policy-table")
def train_policy_table(request: Request) -> dict:
    container = get_container(request)
    return container.models.train_policy_table()


@router.get("/models")
def list_models(request: Request) -> dict:
    container = get_container(request)
    return {"models": container.query_bus.list_models()}


@router.post("/models/{model_id}/evaluate")
def evaluate_model(model_id: str, request: Request) -> dict:
    container = get_container(request)
    try:
        return container.models.evaluate_model_on_spot_packs(model_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="model not found") from exc


@router.post("/datasets/build")
def build_dataset(request: Request, dataset_name: str = "master_v1") -> dict:
    container = get_container(request)
    return container.datasets.build_master_dataset(dataset_name=dataset_name)


@router.get("/datasets")
def list_datasets(request: Request) -> dict:
    container = get_container(request)
    return {"datasets": container.query_bus.list_datasets()}


@router.post("/models/{model_id}/evaluate-dataset/{dataset_id}")
def evaluate_model_dataset(
    model_id: str,
    dataset_id: str,
    request: Request,
    split: str | None = None,
) -> dict:
    container = get_container(request)
    try:
        return container.evaluation.evaluate_model(model_id, dataset_id, split=split)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="dataset not found") from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="model not found") from exc


@router.post("/models/{model_id}/calibrate/{dataset_id}")
def calibrate_model(
    model_id: str,
    dataset_id: str,
    request: Request,
    split: str | None = None,
) -> dict:
    container = get_container(request)
    try:
        return container.calibration.calibrate(model_id, dataset_id, split=split)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="dataset not found") from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="model not found") from exc


@router.get("/solver/catalog")
def solver_catalog() -> dict:
    return {
        "labeler": "solver_like_v1_rule",
        "notes": "offline heuristic labeler until true solver integration",
        "capability_classes": {
            "solver_like_v1_rule": "heuristic",
            "external_solver_adapter_default": "mock",
        },
        "supported_contract": ["mock", "heuristic", "real"],
    }


@router.get("/external-solver/catalog")
def external_solver_catalog(request: Request) -> dict:
    container = get_container(request)
    return container.external_solver.catalog()


@router.post("/external-solver/{solver_name}/compare/{spot_id}")
def compare_external_solver(solver_name: str, spot_id: str, request: Request) -> dict:
    container = get_container(request)
    try:
        return container.external_solver.compare_spot_pack(solver_name, spot_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail="solver or spot pack not found"
        ) from exc
