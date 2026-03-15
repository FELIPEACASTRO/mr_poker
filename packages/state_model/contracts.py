from __future__ import annotations

from pydantic import BaseModel, Field

from packages.common.types import ActionType, Street


class ActionLog(BaseModel):
    actor_seat: int
    action_type: ActionType
    amount: int = 0
    street: Street


class SnapshotLog(BaseModel):
    hand_id: str
    street: Street
    pot: int = Field(ge=0)
    to_call: int = Field(ge=0)
    board: list[str] = Field(default_factory=list)
    active_seats: list[int] = Field(default_factory=list)
    legal_actions: list[ActionType] = Field(default_factory=list)
