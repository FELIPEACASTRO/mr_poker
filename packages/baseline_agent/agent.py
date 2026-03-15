from __future__ import annotations

from packages.baseline_agent.contracts import AgentDecision
from packages.baseline_agent.factory import BaselineAgentFactory
from packages.baseline_agent.strategy import (
    DecisionPolicyStrategy,
    EquityEstimatorStrategy,
)
from packages.engine.engine import GameEngine, HandRuntime
from packages.features.poker import derive_baseline_features
from packages.logging_schema.decision_trace import DecisionTrace


class BaselineAgent:
    """Rule-based baseline agent with pluggable equity/policy strategies."""

    def __init__(
        self,
        *,
        equity_estimator: EquityEstimatorStrategy | None = None,
        decision_strategy: DecisionPolicyStrategy | None = None,
    ) -> None:
        self._equity_estimator = (
            equity_estimator or BaselineAgentFactory.create_equity_estimator()
        )
        self._decision_strategy = (
            decision_strategy or BaselineAgentFactory.create_decision_strategy()
        )

    def decide(self, runtime: HandRuntime, engine: GameEngine) -> AgentDecision:
        state = runtime.state
        acting_seat = state.acting_seat
        if acting_seat is None or state.is_terminal:
            raise ValueError("no action available")
        player = state.players[acting_seat]
        legal = set(engine.legal_actions(runtime))
        features = derive_baseline_features(state, player)
        equity = self._equity_estimator.estimate(runtime, features)
        decision = self._decision_strategy.decide(
            state=state,
            features=features,
            equity=equity,
            legal=legal,
        )
        decision.trace = DecisionTrace(
            hand_id=state.hand_id,
            actor_seat=player.seat,
            action_type=decision.action_type,
            amount=decision.amount,
            rationale=decision.rationale,
            street=features.street,
            hole_class=features.hole_class,
            board_texture=features.board_texture,
            pot=features.pot,
            to_call=features.to_call,
            pot_odds=round(features.pot_odds, 4),
            estimated_equity=round(equity, 4),
            legal_actions=sorted(a.value for a in legal),
            notes=[f"spr={features.spr:.2f}"],
            extra={
                "is_pair": features.is_pair,
                "is_suited": features.is_suited,
                "is_connected": features.is_connected,
                "board_size": features.board_size,
            },
        )
        return decision
