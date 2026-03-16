"""Bayesian Range Estimation for opponent hand inference.

Maintains a probability distribution over the 1326 possible starting hand
combos, updating via Bayesian inference as opponent actions are observed.
Uses MAP estimation for point estimates and Thompson sampling for
exploration-exploitation in range-dependent decisions.

Reference: Bayes' Bluff (Southey et al., UAI 2005)
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any

from packages.common.types import ActionType

logger = logging.getLogger(__name__)

# All 169 canonical hand classes
RANKS = "23456789TJQKA"
HAND_CLASSES: list[str] = []
for i in range(13):
    for j in range(i, 13):
        if i == j:
            HAND_CLASSES.append(f"{RANKS[j]}{RANKS[i]}")  # pair
        else:
            HAND_CLASSES.append(f"{RANKS[j]}{RANKS[i]}s")  # suited
            HAND_CLASSES.append(f"{RANKS[j]}{RANKS[i]}o")  # offsuit

# Number of combos per hand class
COMBO_COUNTS: dict[str, int] = {}
for hc in HAND_CLASSES:
    if len(hc) == 2:  # pair
        COMBO_COUNTS[hc] = 6
    elif hc.endswith("s"):
        COMBO_COUNTS[hc] = 4
    else:
        COMBO_COUNTS[hc] = 12

TOTAL_COMBOS = 1326  # sum of all combo counts


@dataclass
class ActionLikelihood:
    """Likelihood model: P(action | hand_class, context).

    Maps (action, context) pairs to likelihood distributions over hand classes.
    Uses simple heuristic model based on hand strength tiers.
    """

    # Hand strength tiers (simplified)
    PREMIUM: set[str] = field(default_factory=lambda: {
        "AA", "KK", "QQ", "JJ", "AKs", "AKo", "AQs",
    })
    STRONG: set[str] = field(default_factory=lambda: {
        "TT", "99", "88", "AJs", "ATs", "AQo", "KQs", "KJs", "QJs",
    })
    MEDIUM: set[str] = field(default_factory=lambda: {
        "77", "66", "55", "AJo", "ATo", "A9s", "A8s", "KQo", "KJo",
        "KTs", "QTs", "JTs", "T9s", "98s", "87s",
    })

    def likelihood(
        self, hand_class: str, action: ActionType, context: str = "preflop"
    ) -> float:
        """Estimate P(action | hand_class, context).

        Returns a likelihood value (not normalized across hands).
        """
        is_premium = hand_class in self.PREMIUM
        is_strong = hand_class in self.STRONG
        is_medium = hand_class in self.MEDIUM
        is_weak = not (is_premium or is_strong or is_medium)

        if context == "preflop":
            return self._preflop_likelihood(action, is_premium, is_strong, is_medium, is_weak)
        return self._postflop_likelihood(action, is_premium, is_strong, is_medium, is_weak)

    def _preflop_likelihood(
        self, action: ActionType,
        premium: bool, strong: bool, medium: bool, weak: bool,
    ) -> float:
        if action == ActionType.RAISE or action == ActionType.ALL_IN:
            if premium:
                return 0.95
            if strong:
                return 0.70
            if medium:
                return 0.30
            return 0.05
        if action == ActionType.CALL:
            if premium:
                return 0.05  # slowplay
            if strong:
                return 0.25
            if medium:
                return 0.50
            return 0.20
        if action == ActionType.FOLD:
            if premium:
                return 0.001
            if strong:
                return 0.05
            if medium:
                return 0.20
            return 0.75
        # CHECK, BET
        return 0.33

    def _postflop_likelihood(
        self, action: ActionType,
        premium: bool, strong: bool, medium: bool, weak: bool,
    ) -> float:
        if action in (ActionType.BET, ActionType.RAISE, ActionType.ALL_IN):
            if premium:
                return 0.85
            if strong:
                return 0.60
            if medium:
                return 0.35
            return 0.15  # bluff frequency
        if action == ActionType.CALL:
            if premium:
                return 0.10
            if strong:
                return 0.30
            if medium:
                return 0.40
            return 0.25
        if action == ActionType.CHECK:
            if premium:
                return 0.05
            if strong:
                return 0.10
            if medium:
                return 0.25
            return 0.40
        if action == ActionType.FOLD:
            if premium:
                return 0.001
            if strong:
                return 0.05
            if medium:
                return 0.20
            return 0.60
        return 0.25


class BayesianRangeEstimator:
    """Bayesian estimator for opponent's hand range.

    Maintains P(hand_class | observed_actions) via sequential Bayesian updates.
    Starts with a uniform prior and narrows as actions are observed.
    """

    def __init__(self, *, seed: int = 42) -> None:
        self.rng = random.Random(seed)
        self.likelihood_model = ActionLikelihood()
        # Prior: uniform over hand classes, weighted by combo count
        self._log_posterior: dict[str, float] = {
            hc: math.log(COMBO_COUNTS[hc] / TOTAL_COMBOS) for hc in HAND_CLASSES
        }
        self._observations: int = 0

    def reset(self) -> None:
        """Reset to uniform prior."""
        self._log_posterior = {
            hc: math.log(COMBO_COUNTS[hc] / TOTAL_COMBOS) for hc in HAND_CLASSES
        }
        self._observations = 0

    def update(self, action: ActionType, context: str = "preflop") -> None:
        """Update posterior given an observed action.

        Applies Bayes' rule: P(hand|action) ∝ P(action|hand) * P(hand)
        Using log-space for numerical stability.
        """
        for hc in HAND_CLASSES:
            likelihood = self.likelihood_model.likelihood(hc, action, context)
            self._log_posterior[hc] += math.log(max(likelihood, 1e-10))

        # Normalize in log-space
        max_log = max(self._log_posterior.values())
        log_sum = max_log + math.log(
            sum(math.exp(v - max_log) for v in self._log_posterior.values())
        )
        for hc in HAND_CLASSES:
            self._log_posterior[hc] -= log_sum

        self._observations += 1

    def posterior(self) -> dict[str, float]:
        """Get the full posterior distribution P(hand_class | observations).

        Returns probabilities (not log-probabilities).
        """
        max_log = max(self._log_posterior.values())
        raw = {hc: math.exp(v - max_log) for hc, v in self._log_posterior.items()}
        total = sum(raw.values())
        return {hc: v / total for hc, v in raw.items()}

    def map_estimate(self, top_k: int = 10) -> list[tuple[str, float]]:
        """Maximum A Posteriori: return the top-k most likely hand classes."""
        post = self.posterior()
        sorted_hands = sorted(post.items(), key=lambda x: x[1], reverse=True)
        return sorted_hands[:top_k]

    def thompson_sample(self) -> str:
        """Thompson sampling: sample a hand class from the posterior.

        Useful for exploration-exploitation in decision making.
        """
        post = self.posterior()
        hands = list(post.keys())
        weights = [post[h] for h in hands]
        return self.rng.choices(hands, weights=weights, k=1)[0]

    def range_equity_estimate(self, top_pct: float = 0.5) -> float:
        """Estimate opponent's range strength as a percentile.

        Returns a value in [0, 1] where 1.0 = all premium hands.
        """
        post = self.posterior()
        premium_prob = sum(
            post.get(hc, 0.0) for hc in self.likelihood_model.PREMIUM
        )
        strong_prob = sum(
            post.get(hc, 0.0) for hc in self.likelihood_model.STRONG
        )
        medium_prob = sum(
            post.get(hc, 0.0) for hc in self.likelihood_model.MEDIUM
        )

        # Weighted strength score
        strength = premium_prob * 1.0 + strong_prob * 0.7 + medium_prob * 0.4
        return min(1.0, strength)

    def entropy(self) -> float:
        """Shannon entropy of the posterior (lower = more certain)."""
        post = self.posterior()
        return -sum(p * math.log2(p) for p in post.values() if p > 0)

    def confidence(self) -> float:
        """Confidence score in [0, 1]. Higher = more concentrated posterior."""
        # Max entropy for 169 classes = log2(169) ≈ 7.4
        max_entropy = math.log2(len(HAND_CLASSES))
        current = self.entropy()
        return max(0.0, 1.0 - current / max_entropy)
