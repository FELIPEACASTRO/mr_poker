from __future__ import annotations

import random
from dataclasses import dataclass, field

from packages.common.types import ActionType


@dataclass
class ActionDistribution:
    """Probability distribution over actions."""

    probabilities: dict[ActionType, float] = field(default_factory=dict)

    def sample(self, rng: random.Random | None = None) -> ActionType:
        """Sample an action from the distribution."""
        if not self.probabilities:
            raise ValueError("empty distribution")
        r = (rng or random.Random()).random()
        cumulative = 0.0
        for action, prob in self.probabilities.items():
            cumulative += prob
            if r <= cumulative:
                return action
        # Fallback to last action (floating point edge case)
        return list(self.probabilities.keys())[-1]

    def normalize(self) -> ActionDistribution:
        """Return a normalized copy where probabilities sum to 1."""
        total = sum(self.probabilities.values())
        if total == 0:
            return ActionDistribution(probabilities={})
        return ActionDistribution(
            probabilities={k: v / total for k, v in self.probabilities.items()}
        )

    @property
    def dominant_action(self) -> ActionType:
        """Return the most probable action."""
        if not self.probabilities:
            raise ValueError("empty distribution")
        return max(self.probabilities, key=self.probabilities.get)  # type: ignore[arg-type]

    def entropy(self) -> float:
        """Shannon entropy of the distribution."""
        import math
        return -sum(
            p * math.log2(p) for p in self.probabilities.values() if p > 0
        )


class MixedStrategy:
    """Strategy that blends exploit and GTO-like play via regret matching."""

    def __init__(
        self,
        exploit_weight: float = 0.7,
        seed: int = 42,
    ) -> None:
        self.exploit_weight = max(0.0, min(1.0, exploit_weight))
        self.rng = random.Random(seed)
        self._cumulative_regret: dict[str, dict[ActionType, float]] = {}
        self._strategy_sum: dict[str, dict[ActionType, float]] = {}

    def compute_strategy(
        self,
        info_set_key: str,
        legal_actions: set[ActionType],
    ) -> ActionDistribution:
        """Compute current strategy via regret matching."""
        if info_set_key not in self._cumulative_regret:
            # Uniform strategy for unseen info sets
            n = len(legal_actions)
            return ActionDistribution(
                probabilities={a: 1.0 / n for a in legal_actions}
            )

        regrets = self._cumulative_regret[info_set_key]
        positive_regrets = {
            a: max(0.0, regrets.get(a, 0.0)) for a in legal_actions
        }
        total = sum(positive_regrets.values())

        if total > 0:
            probs = {a: r / total for a, r in positive_regrets.items()}
        else:
            n = len(legal_actions)
            probs = {a: 1.0 / n for a in legal_actions}

        return ActionDistribution(probabilities=probs)

    def update_regrets(
        self,
        info_set_key: str,
        action_utilities: dict[ActionType, float],
        chosen_action: ActionType,
    ) -> None:
        """Update cumulative regret after observing utilities."""
        if info_set_key not in self._cumulative_regret:
            self._cumulative_regret[info_set_key] = {}

        chosen_utility = action_utilities.get(chosen_action, 0.0)
        for action, utility in action_utilities.items():
            regret = utility - chosen_utility
            current = self._cumulative_regret[info_set_key].get(action, 0.0)
            self._cumulative_regret[info_set_key][action] = current + regret

    def blend(
        self,
        exploit_dist: ActionDistribution,
        gto_dist: ActionDistribution,
        legal_actions: set[ActionType],
    ) -> ActionDistribution:
        """Blend exploit and GTO strategies with configured weight."""
        blended: dict[ActionType, float] = {}
        for action in legal_actions:
            exploit_p = exploit_dist.probabilities.get(action, 0.0)
            gto_p = gto_dist.probabilities.get(action, 0.0)
            blended[action] = (
                self.exploit_weight * exploit_p + (1 - self.exploit_weight) * gto_p
            )
        return ActionDistribution(probabilities=blended).normalize()
