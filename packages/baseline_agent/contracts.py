from __future__ import annotations

from dataclasses import dataclass

from packages.common.types import ActionType
from packages.logging_schema.decision_trace import DecisionTrace


@dataclass
class AgentDecision:
    action_type: ActionType
    amount: int
    rationale: str
    trace: DecisionTrace | None = None
