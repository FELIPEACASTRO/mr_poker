from __future__ import annotations

from typing import Protocol

from packages.baseline_agent.contracts import AgentDecision
from packages.common.types import ActionType
from packages.engine.engine import HandRuntime
from packages.engine.models import HandState
from packages.equity.monte_carlo import estimate_equity
from packages.evaluator.hands import best_hand_rank
from packages.features.poker import BaselineFeatures


class EquityEstimatorStrategy(Protocol):
    def estimate(self, runtime: HandRuntime, features: BaselineFeatures) -> float: ...


class DecisionPolicyStrategy(Protocol):
    def decide(
        self,
        *,
        state: HandState,
        features: BaselineFeatures,
        equity: float,
        legal: set[ActionType],
    ) -> AgentDecision: ...


class MonteCarloEquityEstimator:
    def estimate(self, runtime: HandRuntime, features: BaselineFeatures) -> float:
        state = runtime.state
        acting_seat = state.acting_seat
        if acting_seat is None:
            raise ValueError("no acting seat to estimate equity")
        player = state.players[acting_seat]
        board_key = "".join(sorted(str(card) for card in state.board))
        street_key = state.street.value
        seed_material = f"{state.hand_id}|{player.seat}|{street_key}|{board_key}"
        seed = sum((index + 1) * ord(char) for index, char in enumerate(seed_material))
        samples = (
            8 if features.board_size == 0 else 10 if features.board_size <= 3 else 15
        )
        return estimate_equity(
            player.hole_cards, state.board, samples=samples, seed=seed
        )


class RuleBasedDecisionStrategy:
    def decide(
        self,
        *,
        state: HandState,
        features: BaselineFeatures,
        equity: float,
        legal: set[ActionType],
    ) -> AgentDecision:
        acting_seat = state.acting_seat
        if acting_seat is None:
            raise ValueError("no acting seat to decide")

        player = state.players[acting_seat]
        max_raise_to = player.invested_this_round + player.stack
        short_stack = features.stack <= 12 * state.big_blind
        premium = features.is_pair and features.high_rank >= 10
        strong_broadway = features.high_rank >= 13 and features.low_rank >= 11
        playable = (
            premium
            or strong_broadway
            or (features.is_suited and features.high_rank >= 11)
            or (features.is_connected and features.high_rank >= 10)
        )

        if features.board_size == 0:
            return self._decide_preflop(
                state=state,
                features=features,
                equity=equity,
                legal=legal,
                max_raise_to=max_raise_to,
                short_stack=short_stack,
                premium=premium,
                playable=playable,
            )

        return self._decide_postflop(
            state=state,
            features=features,
            equity=equity,
            legal=legal,
            max_raise_to=max_raise_to,
            short_stack=short_stack,
            acting_seat=acting_seat,
        )

    def _decide_preflop(
        self,
        *,
        state: HandState,
        features: BaselineFeatures,
        equity: float,
        legal: set[ActionType],
        max_raise_to: int,
        short_stack: bool,
        premium: bool,
        playable: bool,
    ) -> AgentDecision:
        if features.to_call == 0:
            if short_stack and equity >= 0.62 and ActionType.ALL_IN in legal:
                return AgentDecision(
                    ActionType.ALL_IN,
                    features.stack,
                    "push profitable short-stack opening range",
                )
            if equity >= 0.61 and ActionType.BET in legal:
                return AgentDecision(
                    ActionType.BET,
                    max(state.big_blind * 3, 6),
                    "open aggressively with high-equity range",
                )
            if playable and ActionType.BET in legal:
                return AgentDecision(
                    ActionType.BET,
                    max(int(state.big_blind * 2.5), 4),
                    "open playable range",
                )
            return AgentDecision(ActionType.CHECK, 0, "check weakest opening range")

        if short_stack and equity >= 0.64 and ActionType.ALL_IN in legal:
            return AgentDecision(
                ActionType.ALL_IN,
                features.stack,
                "jam profitable short-stack defend range",
            )
        if (premium or equity >= 0.67) and ActionType.RAISE in legal:
            target = max(
                state.min_raise_to or 0, state.current_bet + state.big_blind * 2
            )
            target = min(target, max_raise_to)
            if target >= (state.min_raise_to or target) and target > state.current_bet:
                return AgentDecision(
                    ActionType.RAISE, target, "re-raise strong pre-flop equity"
                )
            if ActionType.ALL_IN in legal and max_raise_to > state.current_bet:
                return AgentDecision(
                    ActionType.ALL_IN,
                    features.stack,
                    "convert short raise into all-in pressure",
                )
        if ActionType.CALL in legal and equity >= max(features.pot_odds + 0.08, 0.42):
            return AgentDecision(
                ActionType.CALL,
                features.to_call,
                "defend with sufficient equity over pot odds",
            )
        if ActionType.FOLD in legal:
            return AgentDecision(
                ActionType.FOLD, 0, "fold low-equity pre-flop continue range"
            )
        return AgentDecision(ActionType.CHECK, 0, "fallback legal action")

    def _decide_postflop(
        self,
        *,
        state: HandState,
        features: BaselineFeatures,
        equity: float,
        legal: set[ActionType],
        max_raise_to: int,
        short_stack: bool,
        acting_seat: int,
    ) -> AgentDecision:
        hand_rank = best_hand_rank(state.players[acting_seat].hole_cards + state.board)
        made_pair_or_better = hand_rank[0] >= 1
        if features.to_call == 0:
            if short_stack and equity >= 0.65 and ActionType.ALL_IN in legal:
                return AgentDecision(
                    ActionType.ALL_IN,
                    features.stack,
                    "jam short-stack with strong equity",
                )
            if (made_pair_or_better or equity >= 0.60) and ActionType.BET in legal:
                size = max(
                    state.big_blind,
                    int(max(state.pot, 2) * (0.60 if equity >= 0.68 else 0.50)),
                )
                return AgentDecision(
                    ActionType.BET, size, "value/protection bet with equity edge"
                )
            if ActionType.CHECK in legal:
                return AgentDecision(
                    ActionType.CHECK, 0, "check medium or weak equity region"
                )

        if short_stack and equity >= 0.66 and ActionType.ALL_IN in legal:
            return AgentDecision(
                ActionType.ALL_IN,
                features.stack,
                "jam short stack over continuing range",
            )
        if equity >= 0.74 and ActionType.RAISE in legal:
            target = max(
                state.min_raise_to or 0,
                state.current_bet + max(state.big_blind * 2, state.pot // 2),
            )
            target = min(target, max_raise_to)
            if target >= (state.min_raise_to or target) and target > state.current_bet:
                return AgentDecision(
                    ActionType.RAISE,
                    target,
                    "raise for value with strong equity advantage",
                )
            if ActionType.ALL_IN in legal and max_raise_to > state.current_bet:
                return AgentDecision(
                    ActionType.ALL_IN,
                    features.stack,
                    "convert capped raise into all-in value",
                )
        if ActionType.CALL in legal and equity >= features.pot_odds + 0.06:
            return AgentDecision(
                ActionType.CALL, features.to_call, "continue when equity exceeds price"
            )
        if ActionType.FOLD in legal:
            return AgentDecision(
                ActionType.FOLD, 0, "fold when equity is below price threshold"
            )
        return AgentDecision(ActionType.CHECK, 0, "fallback legal action")
