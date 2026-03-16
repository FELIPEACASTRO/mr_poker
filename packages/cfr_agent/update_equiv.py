"""Update-Equivalence Framework for Decision-Time Planning.

Implements the update-equivalence framework from Sokota, Farina, Wu, Hu,
Brown (Meta/CMU, 2023).  Instead of expensive subgame solving at decision
time, this uses online mirror descent (OMD) to refine a blueprint strategy
in real-time.

Key insight: For any decision-time planning algorithm that modifies a
blueprint strategy at a single info set, there exists an equivalent
regret-based update that achieves the same result in O(|A|) time
(linear in the number of actions), compared to O(|A|^d) for full
subgame solving (exponential in subgame depth).

Components:
- MirrorDescentRefiner: refines a blueprint strategy using OMD
- DecisionTimePlanner: orchestrates real-time strategy refinement
- GradientEstimator: estimates policy gradients from limited rollouts

Reference: arxiv.org/abs/2304.13138
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from packages.common.types import ActionType
from packages.strategy.mixed import ActionDistribution


def _softmax(values: dict[ActionType, float]) -> dict[ActionType, float]:
    """Softmax over action values."""
    if not values:
        return {}
    max_v = max(values.values())
    exps = {a: math.exp(v - max_v) for a, v in values.items()}
    total = sum(exps.values())
    if total == 0:
        n = len(values)
        return {a: 1.0 / n for a in values}
    return {a: e / total for a, e in exps.items()}


def _kl_divergence(p: dict[ActionType, float], q: dict[ActionType, float]) -> float:
    """KL(p || q) with safe handling of zeros."""
    kl = 0.0
    for a in p:
        pa = p.get(a, 0.0)
        qa = q.get(a, 1e-10)
        if pa > 1e-10:
            kl += pa * math.log(pa / max(qa, 1e-10))
    return kl


def _entropy_regularized_projection(
    logits: dict[ActionType, float],
    tau: float,
) -> dict[ActionType, float]:
    """Project logits onto simplex with entropy regularization (temperature tau).

    This is the mirror map for the entropy distance generating function,
    which yields the softmax operator: pi(a) = exp(logit(a) / tau) / Z.
    """
    if tau <= 0:
        # Greedy: pick the max
        best = max(logits, key=logits.get)  # type: ignore[arg-type]
        return {a: 1.0 if a == best else 0.0 for a in logits}
    scaled = {a: v / tau for a, v in logits.items()}
    return _softmax(scaled)


@dataclass
class RolloutResult:
    """Result of a single rollout from a decision point."""

    action: ActionType
    utility: float


@dataclass
class GradientEstimate:
    """Estimated policy gradient at a decision point."""

    action_values: dict[ActionType, float] = field(default_factory=dict)
    num_samples: int = 0
    baseline: float = 0.0


class GradientEstimator:
    """Estimates policy gradients from sampled rollouts.

    Uses importance-weighted REINFORCE with baseline subtraction
    to estimate the gradient of expected utility w.r.t. the policy.
    """

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)

    def estimate_from_rollouts(
        self,
        rollouts: list[RolloutResult],
        current_strategy: ActionDistribution,
    ) -> GradientEstimate:
        """Estimate action-value gradient from rollout results.

        Args:
            rollouts: List of (action, utility) from sampled play.
            current_strategy: The policy that generated the rollouts.

        Returns:
            GradientEstimate with per-action value estimates.
        """
        if not rollouts:
            return GradientEstimate()

        # Group rollouts by action
        action_utils: dict[ActionType, list[float]] = {}
        for r in rollouts:
            action_utils.setdefault(r.action, []).append(r.utility)

        # Compute mean value per action
        action_values: dict[ActionType, float] = {}
        for action, utils in action_utils.items():
            action_values[action] = sum(utils) / len(utils)

        # Compute baseline (value under current policy)
        baseline = sum(
            current_strategy.probabilities.get(a, 0.0) * action_values.get(a, 0.0)
            for a in current_strategy.probabilities
        )

        return GradientEstimate(
            action_values=action_values,
            num_samples=len(rollouts),
            baseline=baseline,
        )

    def estimate_from_action_values(
        self,
        action_values: dict[ActionType, float],
    ) -> GradientEstimate:
        """Create gradient estimate directly from known action values.

        Useful when action values are available from CFR or Deep CFR
        without needing rollouts.
        """
        if not action_values:
            return GradientEstimate()

        n = len(action_values)
        baseline = sum(action_values.values()) / n

        return GradientEstimate(
            action_values=dict(action_values),
            num_samples=n,
            baseline=baseline,
        )


class MirrorDescentRefiner:
    """Refines a blueprint strategy using Online Mirror Descent.

    The update rule is:
        logit_t+1(a) = logit_t(a) + eta * advantage_t(a)
        pi_t+1 = softmax(logit_t+1 / tau)

    where advantage(a) = Q(a) - V (action value minus baseline).

    This is equivalent to the "update-equivalence" transformation
    from the paper: any valid decision-time refinement can be
    expressed as a sequence of these OMD updates.

    Args:
        learning_rate: Step size for mirror descent (eta).
        temperature: Entropy regularization (tau). Lower = more greedy.
        max_iterations: Maximum OMD steps per refinement.
        convergence_threshold: Stop if KL(pi_new, pi_old) < this.
    """

    def __init__(
        self,
        *,
        learning_rate: float = 0.5,
        temperature: float = 1.0,
        max_iterations: int = 20,
        convergence_threshold: float = 1e-4,
    ) -> None:
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold

    def refine(
        self,
        blueprint: ActionDistribution,
        gradient: GradientEstimate,
        legal: set[ActionType] | None = None,
    ) -> ActionDistribution:
        """Refine blueprint strategy using mirror descent.

        Args:
            blueprint: The pre-computed blueprint strategy.
            gradient: Estimated policy gradient (action values).
            legal: Legal actions (filters output).

        Returns:
            Refined ActionDistribution.
        """
        actions = legal or set(blueprint.probabilities.keys())
        if not actions or not gradient.action_values:
            return blueprint

        # Initialize logits from blueprint (inverse softmax)
        logits: dict[ActionType, float] = {}
        for a in actions:
            p = blueprint.probabilities.get(a, 1e-6)
            logits[a] = math.log(max(p, 1e-10)) * self.temperature

        # Mirror descent iterations
        prev_probs = dict(blueprint.probabilities)
        for _ in range(self.max_iterations):
            # Compute advantages
            for a in actions:
                advantage = gradient.action_values.get(a, 0.0) - gradient.baseline
                logits[a] += self.learning_rate * advantage

            # Project back to simplex via entropy mirror map
            new_probs = _entropy_regularized_projection(logits, self.temperature)

            # Check convergence
            kl = _kl_divergence(new_probs, prev_probs)
            prev_probs = dict(new_probs)

            if kl < self.convergence_threshold:
                break

        # Filter to legal actions and normalize
        filtered = {a: prev_probs.get(a, 0.0) for a in actions}
        total = sum(filtered.values())
        if total > 0:
            filtered = {a: v / total for a, v in filtered.items()}
        else:
            n = len(actions)
            filtered = {a: 1.0 / n for a in actions}

        return ActionDistribution(probabilities=filtered)


class DecisionTimePlanner:
    """Orchestrates real-time strategy refinement at decision points.

    Combines a blueprint strategy with online rollouts/value estimates
    to produce a refined strategy at each decision point. This is the
    update-equivalence approach to decision-time planning.

    Usage::

        planner = DecisionTimePlanner(blueprint_state=cfr_state)

        # At each decision point:
        refined = planner.plan(
            info_set="AKs|flop|IP",
            legal={ActionType.CHECK, ActionType.BET, ActionType.RAISE},
            action_values={ActionType.CHECK: -0.5, ActionType.BET: 1.2, ActionType.RAISE: 0.8},
        )

        # Or with rollouts:
        refined = planner.plan_from_rollouts(
            info_set="AKs|flop|IP",
            legal=legal_actions,
            rollouts=[RolloutResult(ActionType.BET, 2.5), ...],
        )
    """

    def __init__(
        self,
        *,
        learning_rate: float = 0.3,
        temperature: float = 0.5,
        max_iterations: int = 15,
        convergence_threshold: float = 1e-4,
        seed: int = 42,
    ) -> None:
        self.refiner = MirrorDescentRefiner(
            learning_rate=learning_rate,
            temperature=temperature,
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold,
        )
        self.estimator = GradientEstimator(seed=seed)
        self._refinement_count = 0

    def plan(
        self,
        blueprint: ActionDistribution,
        legal: set[ActionType],
        action_values: dict[ActionType, float],
    ) -> ActionDistribution:
        """Refine blueprint using known action values.

        This is the fastest path: O(|A| * max_iterations) time.

        Args:
            blueprint: Pre-computed blueprint strategy.
            legal: Legal actions at this decision point.
            action_values: Estimated value of each action.

        Returns:
            Refined strategy.
        """
        gradient = self.estimator.estimate_from_action_values(action_values)
        result = self.refiner.refine(blueprint, gradient, legal)
        self._refinement_count += 1
        return result

    def plan_from_rollouts(
        self,
        blueprint: ActionDistribution,
        legal: set[ActionType],
        rollouts: list[RolloutResult],
    ) -> ActionDistribution:
        """Refine blueprint using sampled rollout results.

        Args:
            blueprint: Pre-computed blueprint strategy.
            legal: Legal actions at this decision point.
            rollouts: Sampled (action, utility) pairs.

        Returns:
            Refined strategy.
        """
        gradient = self.estimator.estimate_from_rollouts(rollouts, blueprint)
        result = self.refiner.refine(blueprint, gradient, legal)
        self._refinement_count += 1
        return result

    @property
    def refinement_count(self) -> int:
        """Number of refinements performed."""
        return self._refinement_count
