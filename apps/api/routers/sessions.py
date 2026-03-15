from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from apps.api.contracts import SessionRunRequest
from apps.api.routers.utils import get_container
from services.cqrs import SessionH2HCommand

router = APIRouter(prefix="/v1/sessions")


@router.post("/h2h")
def run_session(payload: SessionRunRequest, request: Request) -> dict:
    container = get_container(request)
    return container.command_bus.run_session_h2h(
        SessionH2HCommand(
            num_hands=payload.num_hands,
            stacks=payload.stacks,
            seed_base=payload.seed_base,
            session_name=payload.session_name,
        )
    )


@router.get("/{session_id}")
def get_session(session_id: str, request: Request) -> dict:
    container = get_container(request)
    result = container.query_bus.get_session(session_id)
    if result is None:
        raise HTTPException(status_code=404, detail="session not found")
    return result


@router.get("/{session_id}/traces")
def get_session_traces(session_id: str, request: Request) -> dict:
    container = get_container(request)
    traces = container.query_bus.get_session_traces(session_id)
    if not traces:
        session = container.query_bus.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="session not found")
    return {"session_id": session_id, "traces": traces}


@router.get("/{session_id}/analytics")
def get_session_analytics(session_id: str, request: Request) -> dict:
    container = get_container(request)
    try:
        return container.query_bus.get_session_analytics(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="session not found") from exc


@router.get("/{session_id}/export/phh-like")
def export_session(session_id: str, request: Request) -> dict:
    container = get_container(request)
    try:
        return container.exports.export_session(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="session not found") from exc


@router.get("/{session_id}/curriculum")
def get_curriculum(session_id: str, request: Request) -> dict:
    container = get_container(request)
    try:
        return container.curriculum.build_for_session(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="session not found") from exc


@router.get("/{session_id}/opponent-profile")
def get_opponent_profile(session_id: str, request: Request) -> dict:
    container = get_container(request)
    try:
        return container.opponent_profiles.session_profiles(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="session not found") from exc


@router.get("/{session_id}/coach-report")
def get_coach_report(session_id: str, request: Request) -> dict:
    container = get_container(request)
    try:
        return container.coach.session_report(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="session not found") from exc
