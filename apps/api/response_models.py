from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


# --- Pagination ---

class PaginatedResponse(BaseModel, Generic[T]):
    items: list[T]
    total: int
    limit: int
    offset: int
    has_more: bool


# --- System ---

class HealthResponse(BaseModel):
    status: str
    version: str | None = None


class ReadinessComponent(BaseModel):
    name: str
    status: str
    detail: str = ""


class ReadinessResponse(BaseModel):
    status: str
    components: list[ReadinessComponent]


class ServiceInfoResponse(BaseModel):
    service: str
    version: str
    docs: str


# --- Hands ---

class PlayerSnapshot(BaseModel):
    stack: int
    invested_this_round: int
    total_invested: int
    folded: bool
    is_all_in: bool
    hole_cards: list[str]


class ActionSnapshot(BaseModel):
    seat: int
    action: str
    amount: int
    street: str
    note: str = ""


class HandSnapshot(BaseModel):
    hand_id: str
    street: str
    pot: int
    to_call: int
    acting_seat: int | None
    board: list[str]
    is_terminal: bool
    winner_seat: int | None
    players: dict[str, PlayerSnapshot]
    actions: list[ActionSnapshot]


class AutoActResponse(HandSnapshot):
    decision_rationale: str = ""
    decision_trace: dict[str, Any] | None = None
    persisted_trace_count: int = 0


class HandTracesResponse(BaseModel):
    hand_id: str
    traces: list[dict[str, Any]]


# --- Sessions ---

class SessionSummary(BaseModel):
    session_id: str
    session_type: str
    config: dict[str, Any]
    summary: dict[str, Any] | None = None
    created_at: str | None = None


class SessionResponse(BaseModel):
    session_id: str
    session_type: str
    config: dict[str, Any]
    summary: dict[str, Any] | None


class SessionTracesResponse(BaseModel):
    session_id: str
    traces: list[dict[str, Any]]


# --- Benchmarks ---

class BenchmarkResponse(BaseModel):
    benchmark_type: str
    results: dict[str, Any]


class TournamentResponse(BaseModel):
    standings: list[dict[str, Any]]
    matches: list[dict[str, Any]]


# --- ML ---

class ModelListResponse(BaseModel):
    models: list[dict[str, Any]] | dict[str, Any]


class DatasetListResponse(BaseModel):
    datasets: list[dict[str, Any]] | dict[str, Any]


class TrainResponse(BaseModel):
    model_id: str
    accuracy: float
    num_samples: int


class EvaluationResponse(BaseModel):
    model_id: str
    dataset_id: str
    metrics: dict[str, Any]


# --- Tasks ---

class TaskResponse(BaseModel):
    task_id: str
    task_type: str
    status: str
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    result: Any = None
    error: str | None = None
    progress: float = 0.0


# --- Governance ---

class ModelCardResponse(BaseModel):
    model_id: str
    card: dict[str, Any]


class ReleaseGateResponse(BaseModel):
    passed: bool
    details: dict[str, Any]
