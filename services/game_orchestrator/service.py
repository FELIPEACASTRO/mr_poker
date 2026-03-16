from __future__ import annotations

import logging
from typing import Any

from packages.engine import GameEngine, HandRuntime
from packages.persistence import SqliteHandStore
from services.decision_service.service import DecisionService

logger = logging.getLogger(__name__)


class GameOrchestrator:
    """Coordinates the lifecycle of hands and sessions.

    Facade over engine + persistence + decision service.
    """

    def __init__(
        self,
        engine: GameEngine,
        store: SqliteHandStore,
        decision_service: DecisionService,
    ) -> None:
        self.engine = engine
        self.store = store
        self.decision_service = decision_service

    def create_hand(
        self,
        *,
        stacks: tuple[int, ...] = (100, 100),
        button_seat: int = 0,
        seed: int | None = None,
        deck_prefix: list[str] | None = None,
    ) -> dict[str, Any]:
        runtime = self.engine.start_new_hand(
            stacks=stacks,
            button_seat=button_seat,
            seed=seed,
            deck_prefix=deck_prefix,
        )
        snapshot = self.engine.state_snapshot(runtime)
        self.store.create_hand(
            hand_id=runtime.state.hand_id,
            button_seat=button_seat,
            stacks=stacks,
            seed=seed,
            deck_prefix=deck_prefix,
            initial_snapshot=snapshot,
        )
        return {
            "hand_id": runtime.state.hand_id,
            "runtime": runtime,
            "snapshot": snapshot,
        }

    def play_hand_to_completion(
        self,
        runtime: HandRuntime,
        *,
        agent_name: str = "baseline",
    ) -> dict[str, Any]:
        hand_id = runtime.state.hand_id
        actions_taken = 0
        max_actions = 200  # safety limit

        while not runtime.state.is_terminal and actions_taken < max_actions:
            result = self.decision_service.decide_and_apply(
                runtime, agent_name=agent_name
            )
            actor_seat = result["actor_seat"]
            self.store.append_action(
                hand_id=hand_id,
                actor_seat=int(actor_seat) if actor_seat is not None else -1,
                action_type=result["action_type"].value,
                amount=result["amount"],
            )
            if result.get("trace"):
                trace_obj = result["trace"]
                self.store.append_decision_trace(
                    session_id=None,
                    hand_id=hand_id,
                    actor_seat=trace_obj.actor_seat,
                    trace=trace_obj.model_dump(mode="json"),
                )
            actions_taken += 1

        snapshot = self.engine.state_snapshot(runtime)
        self.store.append_snapshot(hand_id=hand_id, snapshot=snapshot, label="final")
        return {
            "hand_id": hand_id,
            "snapshot": snapshot,
            "actions_taken": actions_taken,
            "is_terminal": runtime.state.is_terminal,
        }

    def replay_hand(self, hand_id: str) -> dict[str, Any]:
        from packages.common.types import ActionType

        hand = self.store.get_hand(hand_id)
        if hand is None:
            raise KeyError(hand_id)
        runtime = self.engine.start_new_hand(
            stacks=tuple(hand["stacks"]),
            button_seat=hand["button_seat"],
            hand_id=hand_id,
            seed=hand["seed"],
            deck_prefix=list(hand["deck_prefix"]),
        )
        for row in self.store.get_actions(hand_id):
            self.engine.apply_action(
                runtime, ActionType(row["action_type"]), int(row["amount"])
            )
        return {
            "hand_id": hand_id,
            "snapshot": self.engine.state_snapshot(runtime),
            "runtime": runtime,
        }
