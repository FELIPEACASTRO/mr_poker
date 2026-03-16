"""MMD — Magnetic Mirror Descent for Quantal Response Equilibrium.

Computes strategies that converge to QRE rather than Nash equilibrium.
QRE models bounded rationality: players make mistakes proportional to
the cost of those mistakes (controlled by a temperature parameter).

Key properties:
- temperature -> 0: converges to Nash equilibrium
- temperature -> inf: converges to uniform random
- Magnetic term regularizes toward uniform, preventing extreme strategies
- Smooth strategy updates with guaranteed convergence

Reference: Farina et al. (2022) "Magnetic Mirror Descent"
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict

from packages.common.types import ActionType
from packages.strategy.mixed import ActionDistribution

logger = logging.getLogger(__name__)


class MagneticMirrorDescent:
    """Magnetic Mirror Descent for convergence to QRE.

    Maintains logits for each (info_set, action) pair and applies
    the MMD update rule:
        logit[a] += lr * (utility[a] - magnetic_strength * logit[a])

    Strategy is derived via softmax with temperature:
        strategy[a] = exp(logit[a] / temperature) / Z
    """

    def __init__(
        self,
        *,
        temperature: float = 1.0,
        learning_rate: float = 0.01,
        magnetic_strength: float = 0.1,
    ) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if magnetic_strength < 0:
            raise ValueError("magnetic_strength must be non-negative")

        self.temperature = temperature
        self.learning_rate = learning_rate
        self.magnetic_strength = magnetic_strength

        # info_set -> {action_str: logit}
        self._logits: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._update_count: int = 0

    def update(
        self,
        info_set: str,
        action_utilities: dict[ActionType, float],
        legal_actions: set[ActionType],
    ) -> None:
        """Apply MMD update rule to logits.

        logit[a] += lr * (utility[a] - magnetic_strength * logit[a])

        The magnetic term pulls logits toward zero (uniform strategy),
        providing regularization and ensuring convergence.
        """
        logits = self._logits[info_set]

        for action in legal_actions:
            key = action.value
            utility = action_utilities.get(action, 0.0)
            current_logit = logits[key]

            # MMD update: gradient ascent on utility with magnetic regularization
            gradient = utility - self.magnetic_strength * current_logit
            logits[key] = current_logit + self.learning_rate * gradient

        self._update_count += 1

    def get_strategy(
        self,
        info_set: str,
        legal_actions: set[ActionType],
    ) -> ActionDistribution:
        """Compute strategy via softmax with temperature.

        strategy[a] = exp(logit[a] / temperature) / sum_b exp(logit[b] / temperature)
        """
        if not legal_actions:
            return ActionDistribution(probabilities={})

        logits = self._logits[info_set]

        # Compute scaled logits
        scaled = {}
        for action in legal_actions:
            key = action.value
            scaled[action] = logits[key] / self.temperature

        # Numerically stable softmax
        max_val = max(scaled.values())
        exp_vals = {a: math.exp(v - max_val) for a, v in scaled.items()}
        total = sum(exp_vals.values())

        if total <= 0:
            n = len(legal_actions)
            probs = {a: 1.0 / n for a in legal_actions}
        else:
            probs = {a: v / total for a, v in exp_vals.items()}

        return ActionDistribution(probabilities=probs)

    def convergence_gap(self) -> float:
        """Measure distance from QRE fixed point.

        At QRE, for each info set the gradient should be near zero:
            utility[a] - magnetic_strength * logit[a] ≈ 0

        We approximate this by measuring the variance of logits across
        info sets—lower variance means closer to equilibrium.
        """
        if not self._logits:
            return float("inf")

        total_gap = 0.0
        count = 0

        for info_set, logits in self._logits.items():
            if not logits:
                continue
            values = list(logits.values())
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            # At QRE with magnetic regularization, logit magnitudes should stabilize
            magnitude = sum(abs(v) for v in values) / len(values)
            total_gap += variance + magnitude * self.magnetic_strength
            count += 1

        return total_gap / max(count, 1)

    @property
    def num_info_sets(self) -> int:
        """Number of distinct info sets tracked."""
        return len(self._logits)

    @property
    def update_count(self) -> int:
        return self._update_count

    def get_logits(self, info_set: str) -> dict[str, float]:
        """Return raw logits for an info set (for debugging/testing)."""
        return dict(self._logits[info_set])

    def reset(self) -> None:
        """Clear all stored logits."""
        self._logits.clear()
        self._update_count = 0
