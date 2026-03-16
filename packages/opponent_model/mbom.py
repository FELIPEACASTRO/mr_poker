"""MBOM — Model-Based Opponent Modeling with recursive belief updates.

Implements recursive opponent modeling where we maintain beliefs about:
1. The opponent's strategy (action distribution per situation)
2. The opponent's model of US (what they think we'll do)
3. How the opponent updates their beliefs based on our actions

This enables deeper exploitation by anticipating how the opponent
will react to our adjustments (level-k reasoning).

Components:
- BeliefState: tracks our beliefs about opponent's strategy
- OpponentModel: predicts opponent actions from observations
- RecursiveModeler: level-k reasoning (what does opponent think we think?)

Reference: Albrecht & Stone (2018) "Autonomous Agents Modelling Other Agents"
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field

from packages.common.types import ActionType
from packages.strategy.mixed import ActionDistribution


@dataclass
class BeliefState:
    """Our beliefs about an opponent's strategy.

    Tracks action frequencies per situation type (street + context).
    Bayesian updates incorporate new observations with a learning rate
    that decays as more data is collected.
    """

    # situation_key -> action -> count
    action_counts: dict[str, dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))
    # Total observations per situation
    situation_totals: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    # Prior strength (pseudo-count for Bayesian smoothing)
    prior_strength: float = 2.0

    def observe(self, situation: str, action: ActionType) -> None:
        """Record an observed action in a given situation."""
        self.action_counts[situation][action.value] += 1
        self.situation_totals[situation] += 1

    def predict(self, situation: str, legal: set[ActionType]) -> ActionDistribution:
        """Predict opponent's action distribution for a situation.

        Uses Bayesian smoothing: (count + prior) / (total + prior * |A|).
        """
        counts = self.action_counts.get(situation, {})
        n = self.situation_totals.get(situation, 0)
        n_actions = len(legal)

        probs: dict[ActionType, float] = {}
        for action in legal:
            count = counts.get(action.value, 0)
            probs[action] = (count + self.prior_strength) / (
                n + self.prior_strength * n_actions
            )

        # Normalize
        total = sum(probs.values())
        if total > 0:
            probs = {a: p / total for a, p in probs.items()}

        return ActionDistribution(probabilities=probs)

    @property
    def total_observations(self) -> int:
        return sum(self.situation_totals.values())

    def confidence(self, situation: str) -> float:
        """Confidence in prediction for a situation (0-1 based on sample size)."""
        n = self.situation_totals.get(situation, 0)
        # Saturating function: 30 observations → ~0.75 confidence
        return 1.0 - math.exp(-n / 20.0)


@dataclass
class RecursiveBeliefState:
    """Multi-level belief state for recursive reasoning.

    level 0: Our model of opponent's strategy
    level 1: Our model of what opponent thinks OUR strategy is
    level 2: Our model of what opponent thinks WE think THEIR strategy is
    """

    levels: dict[int, BeliefState] = field(default_factory=dict)
    max_level: int = 2

    def __post_init__(self):
        for level in range(self.max_level + 1):
            if level not in self.levels:
                self.levels[level] = BeliefState()


class OpponentModel:
    """Predicts opponent actions from accumulated observations.

    Maintains per-situation action frequency tables with Bayesian
    smoothing. Adapts faster to recent actions via exponential
    recency weighting.
    """

    def __init__(self, recency_weight: float = 0.1, seed: int = 42) -> None:
        self.beliefs = BeliefState()
        self.recency_weight = recency_weight
        # Recent action log for recency-weighted estimation
        self._recent: list[tuple[str, str]] = []  # (situation, action_value)
        self._max_recent = 100

    def observe(self, situation: str, action: ActionType) -> None:
        """Record an observed opponent action."""
        self.beliefs.observe(situation, action)
        self._recent.append((situation, action.value))
        if len(self._recent) > self._max_recent:
            self._recent.pop(0)

    def predict(self, situation: str, legal: set[ActionType]) -> ActionDistribution:
        """Predict opponent's action distribution.

        Blends long-term frequency with recent frequency for faster adaptation.
        """
        base_pred = self.beliefs.predict(situation, legal)

        # If enough recent data, blend with recency-weighted prediction
        recent_for_situation = [
            a for s, a in self._recent if s == situation
        ]
        if len(recent_for_situation) < 3:
            return base_pred

        # Compute recent frequencies
        recent_counts: dict[str, int] = defaultdict(int)
        for a in recent_for_situation:
            recent_counts[a] += 1
        n_recent = len(recent_for_situation)

        recent_probs: dict[ActionType, float] = {}
        for action in legal:
            recent_probs[action] = recent_counts.get(action.value, 0) / n_recent

        # Blend: (1-w)*base + w*recent
        w = self.recency_weight
        blended: dict[ActionType, float] = {}
        for action in legal:
            blended[action] = (1 - w) * base_pred.probabilities.get(action, 0.0) + w * recent_probs.get(action, 0.0)

        total = sum(blended.values())
        if total > 0:
            blended = {a: p / total for a, p in blended.items()}

        return ActionDistribution(probabilities=blended)

    @property
    def total_observations(self) -> int:
        return self.beliefs.total_observations


class RecursiveModeler:
    """Level-k recursive opponent modeling.

    Level 0: Direct prediction from observations (what will opponent do?)
    Level 1: Anticipate opponent's adjustment to our strategy
    Level 2: Counter-anticipate (what does opponent think we'll adjust to?)

    Each level uses an OpponentModel to track beliefs.

    Args:
        max_level: Maximum recursion depth (0-3 recommended).
        exploitation_discount: Per-level discount (deeper levels = less confident).
    """

    def __init__(
        self,
        max_level: int = 2,
        exploitation_discount: float = 0.7,
        seed: int = 42,
    ) -> None:
        self.max_level = max_level
        self.exploitation_discount = exploitation_discount
        self._models: dict[int, OpponentModel] = {}
        for level in range(max_level + 1):
            self._models[level] = OpponentModel(seed=seed + level)

    def observe_opponent(self, situation: str, action: ActionType) -> None:
        """Record opponent's action at level 0."""
        self._models[0].observe(situation, action)

    def observe_our_action(self, situation: str, action: ActionType) -> None:
        """Record our action (for opponent's model of us at level 1)."""
        if 1 in self._models:
            self._models[1].observe(situation, action)

    def observe_opponent_adjustment(self, situation: str, action: ActionType) -> None:
        """Record opponent's adjusted action (for level 2 modeling)."""
        if 2 in self._models:
            self._models[2].observe(situation, action)

    def predict(
        self,
        situation: str,
        legal: set[ActionType],
        level: int | None = None,
    ) -> ActionDistribution:
        """Predict opponent's action at the given reasoning level.

        Args:
            situation: Situation key (e.g., "flop_cbet_IP").
            legal: Legal actions.
            level: Reasoning level (None = use best available).

        Returns:
            Predicted action distribution.
        """
        if level is None:
            level = self._best_level(situation)

        level = min(level, self.max_level)
        if level not in self._models:
            level = 0

        return self._models[level].predict(situation, legal)

    def exploitation_weight(self, situation: str) -> float:
        """How much should we exploit based on confidence at each level?

        Higher levels contribute less (discounted), and only if they
        have enough observations.
        """
        weight = 0.0
        for level in range(self.max_level + 1):
            model = self._models[level]
            conf = model.beliefs.confidence(situation)
            level_weight = conf * (self.exploitation_discount ** level)
            weight = max(weight, level_weight)
        return min(1.0, weight)

    def estimated_thinking_level(self) -> int:
        """Estimate opponent's thinking level from observation patterns.

        If level-1 predictions are significantly better than level-0
        at predicting actual opponent behavior, the opponent is
        likely level-1+ (adjusting to us).
        """
        # Simple heuristic: if opponent has few observations, assume level 0
        obs_0 = self._models[0].total_observations
        if obs_0 < 20:
            return 0

        obs_1 = self._models.get(1, OpponentModel()).total_observations
        if obs_1 < 10:
            return 0

        # If opponent has recorded adjustments at level 2, they're adapting
        obs_2 = self._models.get(2, OpponentModel()).total_observations
        if obs_2 > 15:
            return 2

        if obs_1 > 20:
            return 1

        return 0

    def _best_level(self, situation: str) -> int:
        """Choose the best reasoning level for this situation."""
        best = 0
        for level in range(self.max_level + 1):
            conf = self._models[level].beliefs.confidence(situation)
            if conf > 0.3:  # Enough confidence to use this level
                best = level
        return best

    def reset(self) -> None:
        """Clear all models."""
        for model in self._models.values():
            model.beliefs = BeliefState()
            model._recent.clear()
