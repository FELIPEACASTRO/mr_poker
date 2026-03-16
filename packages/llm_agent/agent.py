"""LLM-based poker agent that uses a fine-tuned language model for decisions.

Follows the same pattern as :class:`PolicyTableAgent`: extends
:class:`BaselineAgent`, uses the baseline as fallback when the LLM produces
an illegal or unparseable action.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from packages.baseline_agent import BaselineAgent
from packages.baseline_agent.contracts import AgentDecision
from packages.common.types import ActionType
from packages.dataset_builder.pokerbench_adapter import _parse_action_output
from packages.llm_agent.inference import LLMInference
from packages.llm_agent.prompt import build_pokerbench_prompt

if TYPE_CHECKING:
    from packages.engine.engine import GameEngine, HandRuntime

logger = logging.getLogger(__name__)

_ACTION_MAP: dict[str, ActionType] = {
    "fold": ActionType.FOLD,
    "check": ActionType.CHECK,
    "call": ActionType.CALL,
    "bet": ActionType.BET,
    "raise": ActionType.RAISE,
    "all_in": ActionType.ALL_IN,
    "all-in": ActionType.ALL_IN,
    "allin": ActionType.ALL_IN,
}


class LLMPokerAgent(BaselineAgent):
    """Poker agent powered by a fine-tuned LLM.

    Uses a PokerBench-trained model to select actions. Falls back to the
    baseline rule-based agent when the LLM output is unparseable or illegal.
    """

    def __init__(self, model_id: str, *, quantize: bool = False) -> None:
        super().__init__()
        self.inference = LLMInference(model_id, quantize=quantize)

    def decide(self, runtime: HandRuntime, engine: GameEngine) -> AgentDecision:
        # Get baseline decision first (used as fallback + provides features)
        baseline = super().decide(runtime, engine)

        try:
            prompt = self._build_prompt(runtime, engine, baseline)
            raw_output = self.inference.generate(prompt)
            action_str, amount = _parse_action_output(raw_output)
            action = _ACTION_MAP.get(action_str.lower())

            if action is None:
                logger.debug("LLM output unparseable: %r, using baseline", raw_output)
                return baseline

            legal = set(engine.legal_actions(runtime))
            if action not in legal:
                logger.debug("LLM action %s not legal, using baseline", action)
                return baseline

            # Size the action appropriately
            sized_amount = self._size_action(action, amount, runtime, engine, baseline)

            baseline.action_type = action
            baseline.amount = sized_amount
            baseline.rationale = f"llm:{self.inference.model_id} -> {raw_output}"
            if baseline.trace:
                baseline.trace.action_type = action
                baseline.trace.amount = sized_amount
                baseline.trace.rationale = baseline.rationale
                baseline.trace.notes.append(f"llm_raw={raw_output}")

        except Exception:
            logger.warning("LLM inference failed, using baseline", exc_info=True)

        return baseline

    def _build_prompt(
        self, runtime: HandRuntime, engine: GameEngine, baseline: AgentDecision
    ) -> str:
        """Build a PokerBench-format prompt from the current game state."""
        state = runtime.state
        acting_seat = state.acting_seat
        if acting_seat is None:
            return ""

        player = state.players[acting_seat]
        hole_cards = [str(c) for c in player.hole_cards] if player.hole_cards else []
        board = [str(c) for c in state.community_cards]

        position = "BTN" if acting_seat == state.button_seat else "BB"

        return build_pokerbench_prompt(
            position=position,
            hole_cards=hole_cards,
            board=board,
            pot=float(state.pot),
            street=state.street.value if hasattr(state.street, "value") else str(state.street),
        )

    @staticmethod
    def _size_action(
        action: ActionType,
        llm_amount: int,
        runtime: HandRuntime,
        engine: GameEngine,
        baseline: AgentDecision,
    ) -> int:
        """Determine the correct sizing for the given action."""
        state = runtime.state
        acting_seat = state.acting_seat
        if acting_seat is None:
            return 0

        actor = state.players[acting_seat]

        if action in {ActionType.CHECK, ActionType.FOLD}:
            return 0
        if action == ActionType.CALL:
            return int(state.current_bet - actor.invested_this_round)
        if action == ActionType.ALL_IN:
            return actor.stack
        if action == ActionType.BET:
            # Use LLM amount if reasonable, else half-pot
            if llm_amount > 0:
                return min(llm_amount, actor.stack)
            return max(engine.big_blind, int(max(state.pot, 2) * 0.5))
        if action == ActionType.RAISE:
            if llm_amount > 0:
                candidate = min(llm_amount, actor.invested_this_round + actor.stack)
            else:
                candidate = max(
                    state.min_raise_to or 0,
                    state.current_bet + state.big_blind * 2,
                )
            max_raise = actor.invested_this_round + actor.stack
            return min(candidate, max_raise)

        return baseline.amount
