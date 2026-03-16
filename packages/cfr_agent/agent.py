"""CFR-based poker agent with Deep CFR, opponent modeling, and dynamic exploitation.

Integrates multiple layers of strategy:
1. Tabular CFR average strategy (Nash approximation)
2. Deep CFR neural network for unseen positions
3. Opponent modeling for exploitation adjustments
4. Fast preflop equity for hand strength assessment
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING

from packages.baseline_agent import BaselineAgent
from packages.baseline_agent.contracts import AgentDecision
from packages.cfr_agent.info_set import build_info_set_key
from packages.cfr_agent.trainer import CFRState, _size_action
from packages.common.types import ActionType

if TYPE_CHECKING:
    from packages.cfr_agent.deep_cfr import DeepCFRTrainer
    from packages.engine.engine import GameEngine, HandRuntime
    from packages.opponent_model.classifier import OpponentTracker

logger = logging.getLogger(__name__)


class CFRAgent(BaselineAgent):
    """Advanced poker agent combining CFR, Deep CFR, and opponent modeling.

    Strategy priority:
    1. If info set seen in tabular CFR -> use average strategy
    2. If Deep CFR trainer available -> use neural network prediction
    3. Fall back to baseline agent

    With opponent modeling:
    - Tracks opponent stats across hands
    - Auto-adjusts exploit_blend based on opponent classification
    - Modifies action probabilities based on exploitation adjustments
    """

    def __init__(
        self,
        cfr_state: CFRState,
        *,
        seed: int = 42,
        exploit_blend: float = 0.0,
        deep_cfr: DeepCFRTrainer | None = None,
        opponent_tracker: OpponentTracker | None = None,
    ) -> None:
        super().__init__()
        self.cfr_state = cfr_state
        self.rng = random.Random(seed)
        self.exploit_blend = max(0.0, min(1.0, exploit_blend))
        self.deep_cfr = deep_cfr
        self.opponent_tracker = opponent_tracker

    def decide(self, runtime: HandRuntime, engine: GameEngine) -> AgentDecision:
        baseline = super().decide(runtime, engine)

        state = runtime.state
        if state.acting_seat is None:
            return baseline

        seat = state.acting_seat
        opponent_seat = 1 - seat
        legal = set(engine.legal_actions(runtime))
        info_set = build_info_set_key(state, seat)

        # Layer 1: Get strategy from tabular CFR or Deep CFR
        strategy = self._get_strategy(info_set, legal, state, seat)

        # Layer 2: Dynamic exploit blend from opponent modeling
        effective_blend = self.exploit_blend
        if self.opponent_tracker is not None:
            auto_blend = self.opponent_tracker.compute_exploit_blend(opponent_seat)
            effective_blend = max(effective_blend, auto_blend)

        # Layer 3: Apply exploitation adjustments
        if effective_blend > 0:
            strategy = self._apply_exploitation(
                strategy, baseline, legal, effective_blend, opponent_seat
            )

        # Sample action from final strategy
        sampled_action = strategy.sample(self.rng)

        if sampled_action not in legal:
            logger.debug("CFR sampled illegal %s, using baseline", sampled_action)
            return baseline

        # Size the action
        player = state.players[seat]
        amount = _size_action(sampled_action, state, player, engine)

        # Build rationale
        probs_str = " ".join(
            f"{a.value}={p:.0%}"
            for a, p in sorted(strategy.probabilities.items(), key=lambda x: -x[1])
            if p > 0.01
        )
        entropy = strategy.entropy()

        archetype = "?"
        if self.opponent_tracker is not None:
            archetype = self.opponent_tracker.classify(opponent_seat)

        baseline.action_type = sampled_action
        baseline.amount = amount
        baseline.rationale = f"cfr:{probs_str} H={entropy:.2f} opp={archetype} xb={effective_blend:.2f}"
        if baseline.trace:
            baseline.trace.action_type = sampled_action
            baseline.trace.amount = amount
            baseline.trace.rationale = baseline.rationale
            baseline.trace.notes.append(f"cfr_entropy={entropy:.2f}")
            baseline.trace.notes.append(f"cfr_info_set={info_set}")
            if archetype != "?":
                baseline.trace.notes.append(f"opponent_type={archetype}")

        return baseline

    def _get_strategy(self, info_set: str, legal: set[ActionType], state: object, seat: int):
        """Get strategy from best available source."""
        # Priority 1: Tabular CFR (if info set was seen during training)
        if info_set in self.cfr_state.strategy_sum:
            return self.cfr_state.average_strategy(info_set, legal)

        # Priority 2: Deep CFR neural network
        if self.deep_cfr is not None:
            try:
                from packages.cfr_agent.deep_cfr import extract_features
                feat_vec = extract_features(state, seat).to_vector()
                return self.deep_cfr.get_final_strategy(feat_vec, legal)
            except Exception:
                pass

        # Priority 3: Tabular CFR (uniform for unseen)
        return self.cfr_state.average_strategy(info_set, legal)

    def _apply_exploitation(
        self,
        strategy,
        baseline: AgentDecision,
        legal: set[ActionType],
        blend: float,
        opponent_seat: int,
    ):
        """Apply exploitation adjustments to the GTO strategy."""
        if self.opponent_tracker is None:
            # Simple blend with baseline
            exploit_probs = {a: 0.0 for a in legal}
            exploit_probs[baseline.action_type] = 1.0
            for action in legal:
                ep = exploit_probs.get(action, 0.0)
                gp = strategy.probabilities.get(action, 0.0)
                strategy.probabilities[action] = blend * ep + (1 - blend) * gp
            return strategy.normalize()

        # Advanced: use opponent-specific adjustments
        adjustments = self.opponent_tracker.get_adjustments(opponent_seat)

        adjusted_probs: dict[ActionType, float] = {}
        for action in legal:
            base_prob = strategy.probabilities.get(action, 0.0)

            if action == ActionType.FOLD:
                adjusted_probs[action] = base_prob * adjustments["fold_freq"]
            elif action in {ActionType.BET, ActionType.RAISE, ActionType.ALL_IN}:
                if baseline.action_type in {ActionType.BET, ActionType.RAISE, ActionType.ALL_IN}:
                    adjusted_probs[action] = base_prob * adjustments["value_bet_freq"]
                else:
                    adjusted_probs[action] = base_prob * adjustments["bluff_freq"]
            elif action == ActionType.CALL:
                adjusted_probs[action] = base_prob * adjustments["call_freq"]
            else:
                adjusted_probs[action] = base_prob

        # Blend adjusted with GTO
        for action in legal:
            gto_prob = strategy.probabilities.get(action, 0.0)
            adj_prob = adjusted_probs.get(action, 0.0)
            strategy.probabilities[action] = blend * adj_prob + (1 - blend) * gto_prob

        return strategy.normalize()
