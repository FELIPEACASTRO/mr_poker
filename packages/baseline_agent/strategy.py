from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from packages.baseline_agent.contracts import AgentDecision
from packages.common.types import ActionType
from packages.engine.engine import HandRuntime
from packages.engine.models import HandState
from packages.equity.monte_carlo import estimate_equity
from packages.evaluator.hands import best_hand_rank
from packages.features.poker import BaselineFeatures


@dataclass(frozen=True)
class DecisionConfig:
    """Tunable thresholds for the rule-based decision strategy."""

    # Pre-flop thresholds
    short_stack_bb_multiplier: int = 12
    preflop_open_shove_equity: float = 0.62
    preflop_open_raise_equity: float = 0.61
    preflop_open_raise_bb: float = 3.0
    preflop_open_playable_bb: float = 2.5
    preflop_defend_shove_equity: float = 0.64
    preflop_reraise_equity: float = 0.67
    preflop_call_min_equity: float = 0.42
    preflop_call_equity_margin: float = 0.08

    # Post-flop thresholds
    postflop_shove_equity_no_call: float = 0.65
    postflop_bet_equity: float = 0.60
    postflop_large_bet_equity: float = 0.68
    postflop_large_bet_pot_fraction: float = 0.60
    postflop_small_bet_pot_fraction: float = 0.50
    postflop_shove_equity_facing_bet: float = 0.66
    postflop_raise_equity: float = 0.74
    postflop_call_equity_margin: float = 0.06

    # Hand classification
    premium_rank_threshold: int = 10
    strong_broadway_high: int = 13
    strong_broadway_low: int = 11
    suited_high_threshold: int = 11
    connected_high_threshold: int = 10


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
    def __init__(self, config: DecisionConfig | None = None) -> None:
        self.cfg = config or DecisionConfig()

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

        cfg = self.cfg
        player = state.players[acting_seat]
        max_raise_to = player.invested_this_round + player.stack
        short_stack = features.stack <= cfg.short_stack_bb_multiplier * state.big_blind
        premium = features.is_pair and features.high_rank >= cfg.premium_rank_threshold
        strong_broadway = features.high_rank >= cfg.strong_broadway_high and features.low_rank >= cfg.strong_broadway_low
        playable = (
            premium
            or strong_broadway
            or (features.is_suited and features.high_rank >= cfg.suited_high_threshold)
            or (features.is_connected and features.high_rank >= cfg.connected_high_threshold)
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
        cfg = self.cfg
        if features.to_call == 0:
            if short_stack and equity >= cfg.preflop_open_shove_equity and ActionType.ALL_IN in legal:
                return AgentDecision(
                    ActionType.ALL_IN,
                    features.stack,
                    "push profitable short-stack opening range",
                )
            if equity >= cfg.preflop_open_raise_equity and ActionType.BET in legal:
                return AgentDecision(
                    ActionType.BET,
                    max(int(state.big_blind * cfg.preflop_open_raise_bb), 6),
                    "open aggressively with high-equity range",
                )
            if playable and ActionType.BET in legal:
                return AgentDecision(
                    ActionType.BET,
                    max(int(state.big_blind * cfg.preflop_open_playable_bb), 4),
                    "open playable range",
                )
            return AgentDecision(ActionType.CHECK, 0, "check weakest opening range")

        if short_stack and equity >= cfg.preflop_defend_shove_equity and ActionType.ALL_IN in legal:
            return AgentDecision(
                ActionType.ALL_IN,
                features.stack,
                "jam profitable short-stack defend range",
            )
        if (premium or equity >= cfg.preflop_reraise_equity) and ActionType.RAISE in legal:
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
        if ActionType.CALL in legal and equity >= max(features.pot_odds + cfg.preflop_call_equity_margin, cfg.preflop_call_min_equity):
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
        cfg = self.cfg
        hand_rank = best_hand_rank(state.players[acting_seat].hole_cards + state.board)
        made_pair_or_better = hand_rank[0] >= 1
        if features.to_call == 0:
            if short_stack and equity >= cfg.postflop_shove_equity_no_call and ActionType.ALL_IN in legal:
                return AgentDecision(
                    ActionType.ALL_IN,
                    features.stack,
                    "jam short-stack with strong equity",
                )
            if (made_pair_or_better or equity >= cfg.postflop_bet_equity) and ActionType.BET in legal:
                pot_fraction = cfg.postflop_large_bet_pot_fraction if equity >= cfg.postflop_large_bet_equity else cfg.postflop_small_bet_pot_fraction
                size = max(
                    state.big_blind,
                    int(max(state.pot, 2) * pot_fraction),
                )
                return AgentDecision(
                    ActionType.BET, size, "value/protection bet with equity edge"
                )
            if ActionType.CHECK in legal:
                return AgentDecision(
                    ActionType.CHECK, 0, "check medium or weak equity region"
                )

        if short_stack and equity >= cfg.postflop_shove_equity_facing_bet and ActionType.ALL_IN in legal:
            return AgentDecision(
                ActionType.ALL_IN,
                features.stack,
                "jam short stack over continuing range",
            )
        if equity >= cfg.postflop_raise_equity and ActionType.RAISE in legal:
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
        if ActionType.CALL in legal and equity >= features.pot_odds + cfg.postflop_call_equity_margin:
            return AgentDecision(
                ActionType.CALL, features.to_call, "continue when equity exceeds price"
            )
        if ActionType.FOLD in legal:
            return AgentDecision(
                ActionType.FOLD, 0, "fold when equity is below price threshold"
            )
        return AgentDecision(ActionType.CHECK, 0, "fallback legal action")
