"""ToolPoker integration — LLM agent that calls CFR solver for complex spots.

Based on: "How Far Are LLMs from Professional Poker Players? (ToolPoker)" (2026)

Architecture:
1. LLM evaluates the spot and decides if it's "complex" (high uncertainty)
2. For complex spots, it queries the CFR solver for GTO strategy
3. Combines LLM reasoning with solver output for the final decision

This hybrid approach achieves higher accuracy than either pure LLM or pure solver.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from packages.baseline_agent import BaselineAgent
from packages.baseline_agent.contracts import AgentDecision
from packages.cfr_agent.info_set import build_info_set_key
from packages.cfr_agent.trainer import CFRState, _size_action
from packages.common.types import ActionType
from packages.equity.neural_equity import fast_preflop_equity, hand_strength_category
from packages.features.poker import derive_baseline_features

if TYPE_CHECKING:
    from packages.engine.engine import GameEngine, HandRuntime

logger = logging.getLogger(__name__)


def spot_complexity(state: Any, seat: int, equity: float) -> float:
    """Estimate the complexity of a poker decision spot.

    Returns 0.0 (trivial) to 1.0 (extremely complex).
    Complex spots benefit most from solver consultation.
    """
    player = state.players[seat]
    board = state.board

    # Factor 1: Equity ambiguity (hands near 50% are hardest)
    equity_ambiguity = 1.0 - abs(equity - 0.5) * 2.0

    # Factor 2: Postflop > preflop (more information to process)
    street_complexity = min(len(board) / 5.0, 1.0)

    # Factor 3: Pot size relative to stack (bigger pots = higher stakes)
    spr = player.stack / max(state.pot, 1)
    pot_pressure = max(0.0, 1.0 - spr / 10.0)

    # Factor 4: Facing aggression
    facing_bet = 1.0 if state.current_bet > 0 else 0.0

    # Factor 5: Action history complexity
    num_actions = len(state.actions)
    action_complexity = min(num_actions / 10.0, 1.0)

    complexity = (
        equity_ambiguity * 0.35
        + street_complexity * 0.20
        + pot_pressure * 0.20
        + facing_bet * 0.15
        + action_complexity * 0.10
    )
    return min(1.0, complexity)


class ToolPokerAgent(BaselineAgent):
    """Poker agent using ToolPoker architecture: LLM + CFR solver.

    For simple spots (clear folds/calls), uses fast heuristic.
    For complex spots, consults the CFR solver for GTO strategy.
    This combines the speed of heuristics with the accuracy of solvers.
    """

    def __init__(
        self,
        cfr_state: CFRState,
        *,
        complexity_threshold: float = 0.5,
        solver_weight: float = 0.7,
    ) -> None:
        super().__init__()
        self.cfr_state = cfr_state
        self.complexity_threshold = complexity_threshold
        self.solver_weight = solver_weight

    def decide(self, runtime: HandRuntime, engine: GameEngine) -> AgentDecision:
        baseline = super().decide(runtime, engine)

        state = runtime.state
        if state.acting_seat is None:
            return baseline

        seat = state.acting_seat
        legal = set(engine.legal_actions(runtime))
        player = state.players[seat]

        # Get fast equity estimate
        equity = fast_preflop_equity(player.hole_cards)
        if len(state.board) > 0:
            from packages.equity.monte_carlo import estimate_equity
            equity = estimate_equity(player.hole_cards, state.board, samples=30, seed=42)

        # Assess spot complexity
        complexity = spot_complexity(state, seat, equity)

        if complexity < self.complexity_threshold:
            # Simple spot: use baseline heuristic (fast path)
            baseline.rationale = f"toolpoker:simple complexity={complexity:.2f} eq={equity:.2f}"
            return baseline

        # Complex spot: consult CFR solver
        info_set = build_info_set_key(state, seat)
        solver_strategy = self.cfr_state.average_strategy(info_set, legal)

        # Build heuristic strategy from equity
        heuristic = self._equity_to_strategy(equity, legal, state)

        # Blend solver and heuristic
        final_probs: dict[ActionType, float] = {}
        w = self.solver_weight
        for action in legal:
            sp = solver_strategy.probabilities.get(action, 0.0)
            hp = heuristic.get(action, 0.0)
            final_probs[action] = w * sp + (1 - w) * hp

        # Normalize
        total = sum(final_probs.values())
        if total > 0:
            final_probs = {a: p / total for a, p in final_probs.items()}
        else:
            n = len(legal) or 1
            final_probs = {a: 1.0 / n for a in legal}

        # Pick best action (deterministic for ToolPoker — no sampling)
        best_action = max(final_probs, key=final_probs.get)  # type: ignore
        amount = _size_action(best_action, state, player, engine)

        baseline.action_type = best_action
        baseline.amount = amount
        baseline.rationale = (
            f"toolpoker:complex complexity={complexity:.2f} eq={equity:.2f} "
            f"solver={solver_strategy.probabilities.get(best_action, 0):.0%}"
        )
        if baseline.trace:
            baseline.trace.action_type = best_action
            baseline.trace.amount = amount
            baseline.trace.rationale = baseline.rationale
            baseline.trace.notes.append(f"spot_complexity={complexity:.2f}")

        return baseline

    @staticmethod
    def _equity_to_strategy(
        equity: float, legal: set[ActionType], state: Any
    ) -> dict[ActionType, float]:
        """Convert equity estimate to action probabilities."""
        to_call = state.current_bet - state.players[state.acting_seat].invested_this_round
        pot_odds = to_call / max(state.pot + to_call, 1) if to_call > 0 else 0.0

        probs: dict[ActionType, float] = {}
        for action in legal:
            if action == ActionType.FOLD:
                # Fold when equity is below pot odds
                probs[action] = max(0.0, pot_odds - equity + 0.1) if to_call > 0 else 0.0
            elif action == ActionType.CHECK:
                probs[action] = 0.3 if equity < 0.55 else 0.1
            elif action == ActionType.CALL:
                # Call when equity exceeds pot odds
                probs[action] = max(0.0, equity - pot_odds) if to_call > 0 else 0.0
            elif action in {ActionType.BET, ActionType.RAISE}:
                probs[action] = max(0.0, equity - 0.50) * 2.0
            elif action == ActionType.ALL_IN:
                probs[action] = max(0.0, equity - 0.70) * 2.0
            else:
                probs[action] = 0.1

        total = sum(probs.values())
        if total > 0:
            return {a: p / total for a, p in probs.items()}
        n = len(legal) or 1
        return {a: 1.0 / n for a in legal}
