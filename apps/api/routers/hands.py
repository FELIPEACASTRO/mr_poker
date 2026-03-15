from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from apps.api.contracts import ActionRequest, BatchGenerateRequest, NewHandRequest
from apps.api.routers.utils import get_container

router = APIRouter(prefix="/v1/hands")


@router.post("/new")
def new_hand(payload: NewHandRequest, request: Request) -> dict:
    container = get_container(request)
    runtime = container.engine.start_new_hand(
        stacks=payload.stacks,
        button_seat=payload.button_seat,
        seed=payload.seed,
        deck_prefix=payload.deck_prefix,
    )
    container.runtimes[runtime.state.hand_id] = runtime
    snapshot = container.engine.state_snapshot(runtime)
    container.store.create_hand(
        hand_id=runtime.state.hand_id,
        button_seat=payload.button_seat,
        stacks=payload.stacks,
        seed=payload.seed,
        deck_prefix=payload.deck_prefix,
        initial_snapshot=snapshot,
    )
    return snapshot


@router.get("/{hand_id}")
def get_hand(hand_id: str, request: Request) -> dict:
    container = get_container(request)
    runtime = container.runtimes.get(hand_id)
    if runtime is not None:
        return container.engine.state_snapshot(runtime)
    latest = container.store.get_latest_snapshot(hand_id)
    if latest is None:
        raise HTTPException(status_code=404, detail="hand not found")
    return latest["snapshot"]


@router.post("/{hand_id}/actions")
def act(hand_id: str, payload: ActionRequest, request: Request) -> dict:
    container = get_container(request)
    runtime = container.runtimes.get(hand_id)
    if runtime is None:
        raise HTTPException(status_code=404, detail="hand not found")
    try:
        actor_seat = runtime.state.acting_seat
        container.engine.apply_action(runtime, payload.action_type, payload.amount)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    container.store.append_action(
        hand_id=hand_id,
        actor_seat=int(actor_seat) if actor_seat is not None else -1,
        action_type=payload.action_type.value,
        amount=payload.amount,
    )
    snapshot = container.engine.state_snapshot(runtime)
    container.store.append_snapshot(hand_id=hand_id, snapshot=snapshot, label="post_action")
    return snapshot


@router.post("/{hand_id}/auto")
def auto_act(hand_id: str, request: Request) -> dict:
    container = get_container(request)
    runtime = container.runtimes.get(hand_id)
    if runtime is None:
        raise HTTPException(status_code=404, detail="hand not found")
    try:
        decision = container.agent.decide(runtime, container.engine)
        actor_seat = runtime.state.acting_seat
        container.engine.apply_action(runtime, decision.action_type, decision.amount)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    container.store.append_action(
        hand_id=hand_id,
        actor_seat=int(actor_seat) if actor_seat is not None else -1,
        action_type=decision.action_type.value,
        amount=decision.amount,
    )
    if decision.trace is not None:
        container.store.append_decision_trace(
            session_id=None,
            hand_id=hand_id,
            actor_seat=decision.trace.actor_seat,
            trace=decision.trace.model_dump(mode="json"),
        )
    snapshot = container.engine.state_snapshot(runtime)
    container.store.append_snapshot(hand_id=hand_id, snapshot=snapshot, label="post_auto_action")
    response = dict(snapshot)
    response["decision_rationale"] = decision.rationale
    response["decision_trace"] = decision.trace.model_dump(mode="json") if decision.trace else None
    response["persisted_trace_count"] = len(container.store.get_decision_traces(hand_id=hand_id))
    return response


@router.get("/{hand_id}/traces")
def get_hand_traces(hand_id: str, request: Request) -> dict:
    container = get_container(request)
    return {"hand_id": hand_id, "traces": container.store.get_decision_traces(hand_id=hand_id)}


@router.get("/{hand_id}/replay")
def replay_hand(hand_id: str, request: Request) -> dict:
    container = get_container(request)
    try:
        return container.replay.replay_hand(hand_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="hand not found") from exc


@router.get("/{hand_id}/taxonomy")
def get_hand_taxonomy(hand_id: str, request: Request) -> dict:
    container = get_container(request)
    try:
        return container.taxonomy.classify_hand(hand_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="hand not found") from exc


@router.get("/{hand_id}/export/phh-like")
def export_hand(hand_id: str, request: Request) -> dict:
    container = get_container(request)
    try:
        return container.exports.export_hand(hand_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="hand not found") from exc


@router.get("/{hand_id}/spots/normalized")
def normalized_spot(hand_id: str, request: Request) -> dict:
    container = get_container(request)
    try:
        return container.solver_labels.normalized_spot_for_hand(hand_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="hand not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{hand_id}/review")
def review_hand(hand_id: str, request: Request) -> dict:
    container = get_container(request)
    try:
        return container.coach.hand_review(hand_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="hand not found") from exc


@router.post("/batch-generate")
def generate_batch(payload: BatchGenerateRequest, request: Request) -> dict:
    container = get_container(request)
    return container.batch_generation.generate(
        batch_name=payload.batch_name,
        num_hands=payload.num_hands,
        seed_base=payload.seed_base,
        stacks=payload.stacks,
    )
