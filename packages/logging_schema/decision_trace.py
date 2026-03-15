
from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from packages.common.types import ActionType


class DecisionTrace(BaseModel):
    hand_id: str
    actor_seat: int
    action_type: ActionType
    amount: int
    rationale: str
    feature_version: str = 'v1_sprint04'
    street: str
    hole_class: str
    board_texture: str
    pot: int
    to_call: int
    pot_odds: float
    estimated_equity: float
    legal_actions: list[str]
    notes: list[str] = []
    extra: dict[str, Any] = {}
