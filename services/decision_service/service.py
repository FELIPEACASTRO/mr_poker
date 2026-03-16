from __future__ import annotations

import logging
from typing import Any

from packages.baseline_agent import BaselineAgent
from packages.common.types import ActionType
from packages.engine import GameEngine, HandRuntime

logger = logging.getLogger(__name__)


class DecisionService:
    """Extracts decision logic from the router into a dedicated service.

    Supports multiple agent backends and records decision traces.
    """

    def __init__(
        self,
        engine: GameEngine,
        default_agent: BaselineAgent | None = None,
        agents: dict[str, Any] | None = None,
    ) -> None:
        self.engine = engine
        self.default_agent = default_agent or BaselineAgent()
        self.agents: dict[str, Any] = agents or {"baseline": self.default_agent}

    def decide(
        self,
        runtime: HandRuntime,
        *,
        agent_name: str = "baseline",
    ) -> dict[str, Any]:
        agent = self.agents.get(agent_name, self.default_agent)
        decision = agent.decide(runtime, self.engine)
        return {
            "action_type": decision.action_type,
            "amount": decision.amount,
            "rationale": decision.rationale,
            "trace": decision.trace,
            "agent_name": agent_name,
        }

    def decide_and_apply(
        self,
        runtime: HandRuntime,
        *,
        agent_name: str = "baseline",
    ) -> dict[str, Any]:
        result = self.decide(runtime, agent_name=agent_name)
        actor_seat = runtime.state.acting_seat
        self.engine.apply_action(runtime, result["action_type"], result["amount"])
        result["actor_seat"] = actor_seat
        result["snapshot"] = self.engine.state_snapshot(runtime)
        return result

    def apply_manual_action(
        self,
        runtime: HandRuntime,
        action_type: ActionType,
        amount: int,
    ) -> dict[str, Any]:
        actor_seat = runtime.state.acting_seat
        self.engine.apply_action(runtime, action_type, amount)
        snapshot = self.engine.state_snapshot(runtime)
        return {
            "actor_seat": actor_seat,
            "action_type": action_type,
            "amount": amount,
            "snapshot": snapshot,
        }

    def list_agents(self) -> list[str]:
        return list(self.agents.keys())
