from __future__ import annotations

from collections import Counter

from packages.baseline_agent import BaselineAgent
from packages.baseline_agent.contracts import AgentDecision
from packages.common.types import ActionType


class AdaptiveBaselineAgent(BaselineAgent):
    def __init__(self) -> None:
        super().__init__()
        self._seen_actions = Counter()
        self._seen_hands: set[str] = set()

    def decide(self, runtime, engine) -> AgentDecision:
        self._observe(runtime)
        decision = super().decide(runtime, engine)
        aggression = self._aggression_rate()
        # Adapt only on marginal continues.
        if decision.trace is not None:
            eq = float(decision.trace.estimated_equity)
            price = float(decision.trace.pot_odds)
            margin = eq - price
            if aggression >= 0.45 and decision.action_type == ActionType.CALL and margin < 0.10:
                if ActionType.FOLD in engine.legal_actions(runtime):
                    decision.action_type = ActionType.FOLD
                    decision.amount = 0
                    decision.rationale = 'adaptive fold versus aggressive opponent profile'
                    decision.trace.action_type = ActionType.FOLD
                    decision.trace.amount = 0
                    decision.trace.rationale = decision.rationale
                    decision.trace.notes.append('adaptive_profile=aggressive')
            elif aggression < 0.25 and decision.action_type == ActionType.FOLD and margin > -0.02:
                if ActionType.CALL in engine.legal_actions(runtime):
                    decision.action_type = ActionType.CALL
                    decision.amount = int(decision.trace.to_call)
                    decision.rationale = 'adaptive bluff-catch versus passive opponent profile'
                    decision.trace.action_type = ActionType.CALL
                    decision.trace.amount = int(decision.trace.to_call)
                    decision.trace.rationale = decision.rationale
                    decision.trace.notes.append('adaptive_profile=passive')
        return decision

    def _observe(self, runtime) -> None:
        for event in runtime.state.actions:
            hand_key = (runtime.state.hand_id, len(runtime.state.actions))
            # Counter from whole history is okay here because actions repeat deterministically once per call path.
            # We only count unseen action indexes per hand.
        # Use simpler stable summary from current runtime on each call.
        counts = Counter(a.action_type.value for a in runtime.state.actions if a.actor_seat != runtime.state.acting_seat)
        self._seen_actions = counts
        self._seen_hands.add(runtime.state.hand_id)

    def _aggression_rate(self) -> float:
        total = sum(self._seen_actions.values())
        if total == 0:
            return 0.33
        agg = sum(self._seen_actions.get(k, 0) for k in ('bet', 'raise', 'all_in'))
        return agg / total
