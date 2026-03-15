from __future__ import annotations

from pydantic import BaseModel, Field

from packages.common.types import ActionType


class NewHandRequest(BaseModel):
    stacks: tuple[int, int] = (100, 100)
    button_seat: int = 0
    seed: int | None = None
    deck_prefix: list[str] = Field(default_factory=list)


class ActionRequest(BaseModel):
    action_type: ActionType
    amount: int = 0


class SmokeBenchmarkRequest(BaseModel):
    num_hands: int = 50
    stacks: tuple[int, int] = (100, 100)
    seed_base: int = 10_000


class H2HBenchmarkRequest(BaseModel):
    num_matches: int = 3
    hands_per_match: int = 50
    stacks: tuple[int, int] = (100, 100)
    seed_base: int = 20_000


class SessionRunRequest(BaseModel):
    num_hands: int = 100
    stacks: tuple[int, int] = (100, 100)
    seed_base: int = 30_000
    session_name: str = "local_h2h_session"


class AdaptiveBenchmarkRequest(BaseModel):
    num_hands: int = 40
    stacks: tuple[int, int] = (100, 100)
    seed_base: int = 50_000


class TournamentRequest(BaseModel):
    entrants: list[str] = Field(default_factory=lambda: ["baseline", "adaptive"])
    hands_per_match: int = 20
    seed_base: int = 70_000
    stacks: tuple[int, int] = (100, 100)


class BatchGenerateRequest(BaseModel):
    batch_name: str = "baseline_batch"
    num_hands: int = 50
    seed_base: int = 90_000
    stacks: tuple[int, int] = (100, 100)


class ReleaseGateRequest(BaseModel):
    min_accuracy: float = 0.55
    min_rows: int = 10


class RegressionSuiteRequest(BaseModel):
    suite_name: str = "core_regression_v1"
