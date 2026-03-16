from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from apps.api.contracts import RegressionSuiteRequest, ReleaseGateRequest
from apps.api.routers.utils import get_container
from packages.auth.dependencies import require_admin
from packages.auth.models import User

router = APIRouter(prefix="/v1")


@router.post("/governance/model-cards/{model_id}")
def build_model_card(model_id: str, request: Request, user: User = Depends(require_admin)) -> dict:
    container = get_container(request)
    container.audit.log_admin(user.username, "build_model_card", {"model_id": model_id})
    try:
        return container.governance.build_model_card(model_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="model not found") from exc


@router.get("/governance/model-cards")
def list_model_cards(request: Request) -> dict:
    container = get_container(request)
    return {"model_cards": container.governance.list_model_cards()}


@router.post("/release/gate/{model_id}/{dataset_id}")
def evaluate_release_gate(
    model_id: str,
    dataset_id: str,
    payload: ReleaseGateRequest,
    request: Request,
    user: User = Depends(require_admin),
) -> dict:
    container = get_container(request)
    container.audit.log_admin(user.username, "evaluate_release_gate", {"model_id": model_id, "dataset_id": dataset_id})
    try:
        return container.release_gate.evaluate_candidate(
            model_id,
            dataset_id,
            min_accuracy=payload.min_accuracy,
            min_rows=payload.min_rows,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="dataset not found") from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="model not found") from exc


@router.post("/regression/suites/build")
def build_regression_suite(payload: RegressionSuiteRequest, request: Request, user: User = Depends(require_admin)) -> dict:
    container = get_container(request)
    return container.regression_suites.build(suite_name=payload.suite_name)
