from __future__ import annotations

from packages.baseline_agent import BaselineAgent
from packages.baseline_agent.contracts import AgentDecision
from packages.common.types import ActionType
from packages.solver_like import bucketize_spot, normalize_runtime


class PolicyTableAgent(BaselineAgent):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def decide(self, runtime, engine) -> AgentDecision:
        baseline = super().decide(runtime, engine)
        spot = normalize_runtime(runtime, engine)
        spot['estimated_equity'] = float(baseline.trace.estimated_equity) if baseline.trace else 0.0
        spot['bucket_info'] = bucketize_spot(spot)
        pred = self.model.predict(spot['bucket_info']['bucket_key'], fallback=baseline.action_type.value)
        legal = set(engine.legal_actions(runtime))
        action = ActionType(pred['action'])
        if action not in legal:
            return baseline
        amount = baseline.amount
        # Normalize impossible action classes before sizing.
        if action == ActionType.BET and int(spot.get('to_call', 0) or 0) > 0:
            action = ActionType.CALL if ActionType.CALL in legal else (ActionType.RAISE if ActionType.RAISE in legal else action)
        if action == ActionType.BET and int(spot.get('to_call', 0) or 0) == 0 and runtime.state.current_bet > 0 and ActionType.CHECK in legal:
            action = ActionType.CHECK
        if action == ActionType.CALL and int(spot.get('to_call', 0) or 0) == 0 and ActionType.CHECK in legal:
            action = ActionType.CHECK
        if action in {ActionType.CHECK, ActionType.FOLD, ActionType.CALL}:
            amount = 0 if action in {ActionType.CHECK, ActionType.FOLD} else int(spot.get('to_call', 0) or 0)
        elif action == ActionType.BET:
            amount = max(engine.big_blind, int(max(spot.get('pot', 2), 2) * 0.5))
        elif action == ActionType.RAISE:
            state = runtime.state
            actor = state.players[state.acting_seat]
            candidate = max(state.min_raise_to or 0, state.current_bet + state.big_blind * 2)
            max_raise_to = actor.invested_this_round + actor.stack
            amount = min(candidate, max_raise_to)
            if state.min_raise_to is not None and amount < state.min_raise_to:
                # Convert impossible raise into a legal fallback.
                if ActionType.CALL in legal:
                    action = ActionType.CALL
                    amount = int(spot.get('to_call', 0) or 0)
                elif ActionType.ALL_IN in legal:
                    action = ActionType.ALL_IN
                    amount = actor.stack
                else:
                    return baseline
        elif action == ActionType.ALL_IN:
            actor = runtime.state.players[runtime.state.acting_seat]
            amount = actor.stack
        baseline.action_type = action
        baseline.amount = int(amount)
        baseline.rationale = f'policy table action with confidence={pred["confidence"]}'
        if baseline.trace:
            baseline.trace.action_type = action
            baseline.trace.amount = int(amount)
            baseline.trace.rationale = baseline.rationale
            baseline.trace.notes.append(f'policy_model_confidence={pred["confidence"]}')
        return baseline
