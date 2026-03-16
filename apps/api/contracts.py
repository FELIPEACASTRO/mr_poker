from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from packages.common.types import ActionType


class NewHandRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stacks: tuple[int, int] = (100, 100)
    button_seat: int = Field(default=0, ge=0, le=1)
    seed: int | None = None
    deck_prefix: list[str] = Field(default_factory=list)


class ActionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action_type: ActionType
    amount: int = Field(default=0, ge=0)


class SmokeBenchmarkRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    num_hands: int = Field(default=50, ge=1, le=10000)
    stacks: tuple[int, int] = (100, 100)
    seed_base: int = Field(default=10_000, ge=0)


class H2HBenchmarkRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    num_matches: int = Field(default=3, ge=1, le=100)
    hands_per_match: int = Field(default=50, ge=1, le=1000)
    stacks: tuple[int, int] = (100, 100)
    seed_base: int = Field(default=20_000, ge=0)


class SessionRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    num_hands: int = Field(default=100, ge=1, le=10000)
    stacks: tuple[int, int] = (100, 100)
    seed_base: int = Field(default=30_000, ge=0)
    session_name: str = Field(default="local_h2h_session", pattern=r"^[a-zA-Z0-9_-]+$")


class AdaptiveBenchmarkRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    num_hands: int = Field(default=40, ge=1, le=10000)
    stacks: tuple[int, int] = (100, 100)
    seed_base: int = Field(default=50_000, ge=0)


class TournamentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    entrants: list[str] = Field(default_factory=lambda: ["baseline", "adaptive"])
    hands_per_match: int = Field(default=20, ge=1, le=1000)
    seed_base: int = Field(default=70_000, ge=0)
    stacks: tuple[int, int] = (100, 100)


class BatchGenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    batch_name: str = Field(default="baseline_batch", pattern=r"^[a-zA-Z0-9_-]+$")
    num_hands: int = Field(default=50, ge=1, le=10000)
    seed_base: int = Field(default=90_000, ge=0)
    stacks: tuple[int, int] = (100, 100)


class ReleaseGateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    min_accuracy: float = Field(default=0.55, ge=0.0, le=1.0)
    min_rows: int = Field(default=10, ge=1)


class RegressionSuiteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    suite_name: str = Field(default="core_regression_v1", pattern=r"^[a-zA-Z0-9_-]+$")
