
from __future__ import annotations

from packages.baseline_agent.contracts import AgentDecision
from packages.common.types import ActionType
from packages.engine.engine import GameEngine, HandRuntime
from packages.equity import estimate_equity
from packages.evaluator.hands import best_hand_rank
from packages.features.poker import BaselineFeatures, derive_baseline_features
from packages.logging_schema.decision_trace import DecisionTrace


class BaselineAgent:
    """Sprint 04 local agent.

    Still deterministic and lightweight, but stronger than the Sprint 02/03 version.
    It uses:
    - simple pre-flop tiers
    - lightweight Monte Carlo equity for post-flop and close pre-flop spots
    - pot odds checks
    - short-stack aggression rules
    """

    def decide(self, runtime: HandRuntime, engine: GameEngine) -> AgentDecision:
        state = runtime.state
        if state.acting_seat is None or state.is_terminal:
            raise ValueError('no action available')
        player = state.players[state.acting_seat]
        legal = set(engine.legal_actions(runtime))
        features = derive_baseline_features(state, player)
        equity = self._estimate_equity(runtime, features)
        decision = self._decide_with_equity(state=state, features=features, equity=equity, legal=legal)
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
            notes=[f'spr={features.spr:.2f}'],
            extra={
                'is_pair': features.is_pair,
                'is_suited': features.is_suited,
                'is_connected': features.is_connected,
                'board_size': features.board_size,
            },
        )
        return decision

    def _estimate_equity(self, runtime: HandRuntime, features: BaselineFeatures) -> float:
        state = runtime.state
        player = state.players[state.acting_seat]
        # Cheap deterministic seed derived from hand and action count.
        seed = sum(ord(c) for c in state.hand_id) + (len(state.actions) * 97) + player.seat
        samples = 180 if features.board_size == 0 else 240 if features.board_size <= 3 else 320
        return estimate_equity(player.hole_cards, state.board, samples=samples, seed=seed)

    def _decide_with_equity(self, *, state, features: BaselineFeatures, equity: float, legal: set[ActionType]) -> AgentDecision:
        player = state.players[state.acting_seat]
        max_raise_to = player.invested_this_round + player.stack
        short_stack = features.stack <= 12 * state.big_blind
        premium = features.is_pair and features.high_rank >= 10
        strong_broadway = features.high_rank >= 13 and features.low_rank >= 11
        playable = premium or strong_broadway or (features.is_suited and features.high_rank >= 11) or (features.is_connected and features.high_rank >= 10)

        if features.board_size == 0:
            if features.to_call == 0:
                if short_stack and equity >= 0.62 and ActionType.ALL_IN in legal:
                    return AgentDecision(ActionType.ALL_IN, features.stack, 'push profitable short-stack opening range')
                if equity >= 0.61 and ActionType.BET in legal:
                    return AgentDecision(ActionType.BET, max(state.big_blind * 3, 6), 'open aggressively with high-equity range')
                if playable and ActionType.BET in legal:
                    return AgentDecision(ActionType.BET, max(int(state.big_blind * 2.5), 4), 'open playable range')
                return AgentDecision(ActionType.CHECK, 0, 'check weakest opening range')

            # Facing a bet/raise pre-flop.
            if short_stack and equity >= 0.64 and ActionType.ALL_IN in legal:
                return AgentDecision(ActionType.ALL_IN, features.stack, 'jam profitable short-stack defend range')
            if (premium or equity >= 0.67) and ActionType.RAISE in legal:
                target = max(state.min_raise_to or 0, state.current_bet + state.big_blind * 2)
                target = min(target, max_raise_to)
                if target >= (state.min_raise_to or target) and target > state.current_bet:
                    return AgentDecision(ActionType.RAISE, target, 're-raise strong pre-flop equity')
                if ActionType.ALL_IN in legal and max_raise_to > state.current_bet:
                    return AgentDecision(ActionType.ALL_IN, features.stack, 'convert short raise into all-in pressure')
            if ActionType.CALL in legal and equity >= max(features.pot_odds + 0.08, 0.42):
                return AgentDecision(ActionType.CALL, features.to_call, 'defend with sufficient equity over pot odds')
            if ActionType.FOLD in legal:
                return AgentDecision(ActionType.FOLD, 0, 'fold low-equity pre-flop continue range')
            return AgentDecision(ActionType.CHECK, 0, 'fallback legal action')

        # Post-flop branch.
        hand_rank = best_hand_rank(state.players[state.acting_seat].hole_cards + state.board)
        made_pair_or_better = hand_rank[0] >= 1
        if features.to_call == 0:
            if short_stack and equity >= 0.65 and ActionType.ALL_IN in legal:
                return AgentDecision(ActionType.ALL_IN, features.stack, 'jam short-stack with strong equity')
            if (made_pair_or_better or equity >= 0.60) and ActionType.BET in legal:
                size = max(state.big_blind, int(max(state.pot, 2) * (0.60 if equity >= 0.68 else 0.50)))
                return AgentDecision(ActionType.BET, size, 'value/protection bet with equity edge')
            if ActionType.CHECK in legal:
                return AgentDecision(ActionType.CHECK, 0, 'check medium or weak equity region')

        if short_stack and equity >= 0.66 and ActionType.ALL_IN in legal:
            return AgentDecision(ActionType.ALL_IN, features.stack, 'jam short stack over continuing range')
        if equity >= 0.74 and ActionType.RAISE in legal:
            target = max(state.min_raise_to or 0, state.current_bet + max(state.big_blind * 2, state.pot // 2))
            target = min(target, max_raise_to)
            if target >= (state.min_raise_to or target) and target > state.current_bet:
                return AgentDecision(ActionType.RAISE, target, 'raise for value with strong equity advantage')
            if ActionType.ALL_IN in legal and max_raise_to > state.current_bet:
                return AgentDecision(ActionType.ALL_IN, features.stack, 'convert capped raise into all-in value')
        if ActionType.CALL in legal and equity >= features.pot_odds + 0.06:
            return AgentDecision(ActionType.CALL, features.to_call, 'continue when equity exceeds price')
        if ActionType.FOLD in legal:
            return AgentDecision(ActionType.FOLD, 0, 'fold when equity is below price threshold')
        return AgentDecision(ActionType.CHECK, 0, 'fallback legal action')
