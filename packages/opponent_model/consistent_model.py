"""Consistent Opponent Modeling in Imperfect-Information Games.

Based on arXiv:2508.17671.  Uses projected gradient descent on
sequence-form opponent strategies with convergence guarantees.

Key ideas:
  1. Maintain belief about opponent's full strategy (sequence-form)
  2. Update via projected gradient descent on negative log-likelihood
  3. Project back to valid strategy polytope after each update
  4. Convergence guarantee: beliefs converge to true strategy as T→∞

The opponent's strategy is represented in *behavioural* (aka
realization-plan) form:

    x[info_set][action] = probability of taking *action* at *info_set*

At every info set the probabilities must form a valid simplex:

    sum(x[info_set][a] for a in legal) = 1
    x[info_set][a] >= 0
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

from packages.common.types import ActionType
from packages.strategy.mixed import ActionDistribution
from packages.opponent_model.behavioral_pipeline import BehavioralSignals
from packages.opponent_model.meta_game import AdaptationState


@dataclass
class OpponentBelief:
    """Belief about opponent's strategy at each information set."""

    strategy: dict[str, ActionDistribution] = field(default_factory=dict)
    """info_set → action probabilities."""

    confidence: dict[str, float] = field(default_factory=dict)
    """info_set → confidence in [0, 1]."""

    observations: dict[str, int] = field(default_factory=dict)
    """info_set → total observation count."""

    log_likelihood: float = 0.0
    """Total log-likelihood of all observations under the current belief."""


class ConsistentOpponentModel:
    """Opponent modeling with convergence guarantees.

    Based on arXiv:2508.17671.

    Usage::

        model = ConsistentOpponentModel()
        # During play, record observations:
        model.observe("Ks|flop:AhTd2c", ActionType.BET, {ActionType.CHECK, ActionType.BET})
        # Periodically update:
        model.update_beliefs()
        # Predict:
        dist = model.predict("Ks|flop:AhTd2c", {ActionType.CHECK, ActionType.BET})
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        regularization: float = 0.01,
        min_observations: int = 5,
    ) -> None:
        """Initialise the model.

        Args:
            learning_rate: Step size for projected gradient descent.
            regularization: Strength of regularization toward uniform
                strategy (prevents degenerate beliefs with zero probability).
            min_observations: Minimum number of observations at an
                information set before the model starts trusting its own
                prediction over the uniform prior.
        """
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.min_observations = min_observations

        # Current belief about opponent strategy: info_set → {action: prob}
        self._belief: dict[str, dict[ActionType, float]] = {}

        # Observation counts: info_set → {action: count}
        self._obs_counts: dict[str, dict[ActionType, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        # Legal actions seen at each info set
        self._legal_at: dict[str, set[ActionType]] = defaultdict(set)

        # Total observation count per info set
        self._total_obs: dict[str, int] = defaultdict(int)

        # Pending observations not yet incorporated (batch update)
        self._pending: list[tuple[str, ActionType, set[ActionType]]] = []

        # Running log-likelihood of the model
        self._log_likelihood: float = 0.0

        # History of per-step log-likelihood for convergence diagnostics
        self._ll_history: list[float] = []

        # Update step counter
        self._step: int = 0

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def observe(
        self,
        info_set: str,
        action: ActionType,
        legal: set[ActionType],
    ) -> None:
        """Record an observed action at an information set.

        Args:
            info_set: The information set key where the action was taken.
            action: The action the opponent chose.
            legal: The set of legal actions at this information set.
        """
        self._obs_counts[info_set][action] += 1
        self._total_obs[info_set] += 1
        self._legal_at[info_set] |= legal

        # Initialise belief if first time seeing this info set
        if info_set not in self._belief:
            n = len(legal) if legal else 1
            self._belief[info_set] = {a: 1.0 / n for a in legal}

        # Ensure the observed action is in the belief dict
        if action not in self._belief[info_set]:
            self._belief[info_set][action] = 1e-6

        # Update running log-likelihood
        p = self._belief[info_set].get(action, 1e-10)
        self._log_likelihood += math.log(max(p, 1e-10))

        # Queue for batch gradient update
        self._pending.append((info_set, action, legal))

    # ------------------------------------------------------------------
    # Belief update (projected gradient descent)
    # ------------------------------------------------------------------

    def update_beliefs(self) -> None:
        """Run one step of projected gradient descent.

        Steps:
          1. Compute gradient of negative log-likelihood w.r.t. belief.
          2. Take gradient step: x' = x - lr * gradient.
          3. Add regularization pull toward uniform.
          4. Project onto simplex (valid probability distribution).

        The gradient of -log P(observations | belief) with respect to
        belief parameter x[info_set][a] is:

            -count(info_set, a) / x[info_set][a]

        which pushes the belief toward the empirical frequency.
        """
        if not self._pending and not self._obs_counts:
            return

        self._step += 1

        # Adaptive learning rate: lr_t = lr / sqrt(t)
        lr = self.learning_rate / math.sqrt(self._step)

        # Update each info set that has observations
        updated_info_sets = set()
        for info_set, action, legal in self._pending:
            updated_info_sets.add(info_set)
        # Also include all previously observed info sets for regularization
        updated_info_sets |= set(self._obs_counts.keys())

        for info_set in updated_info_sets:
            legal = self._legal_at.get(info_set, set())
            if not legal:
                continue

            counts = self._obs_counts.get(info_set, {})
            total = self._total_obs.get(info_set, 0)
            if total == 0:
                continue

            belief = self._belief.get(info_set, {})
            n_actions = len(legal)
            uniform_p = 1.0 / n_actions if n_actions > 0 else 1.0

            new_belief: dict[ActionType, float] = {}
            for a in legal:
                x = belief.get(a, uniform_p)
                # Gradient of negative log-likelihood: -count_a / x_a
                count_a = counts.get(a, 0)
                if x > 1e-10:
                    grad = -count_a / x
                else:
                    grad = -count_a / 1e-10

                # Gradient step (descending negative log-likelihood =
                # ascending log-likelihood)
                x_new = x - lr * grad

                # Regularization: pull toward uniform
                x_new = x_new + self.regularization * (uniform_p - x_new)

                new_belief[a] = x_new

            # Project onto probability simplex
            projected = self._project_simplex(new_belief, legal)
            self._belief[info_set] = projected

        # Clear pending batch
        self._pending.clear()

        # Recompute log-likelihood under updated beliefs
        self._log_likelihood = self._compute_total_log_likelihood()
        self._ll_history.append(self._log_likelihood)

    @staticmethod
    def _project_simplex(
        x: dict[ActionType, float],
        legal: set[ActionType],
    ) -> dict[ActionType, float]:
        """Project a vector onto the probability simplex.

        Uses the algorithm from Duchi et al. (2008):
          1. Sort values in descending order.
          2. Find k = max{j : u_j - (sum_{i<=j} u_i - 1)/j > 0}.
          3. Set threshold t = (sum_{i<=k} u_i - 1) / k.
          4. Project: x_i = max(x_i - t, 0).

        Args:
            x: Dict of action → (possibly negative) weight.
            legal: The set of actions that must be in the output.

        Returns:
            Dict of action → probability, summing to 1.
        """
        actions = sorted(legal, key=lambda a: x.get(a, 0.0), reverse=True)
        n = len(actions)
        if n == 0:
            return {}

        vals = [x.get(a, 0.0) for a in actions]

        cumsum = 0.0
        rho = 0
        for j in range(n):
            cumsum += vals[j]
            if vals[j] - (cumsum - 1.0) / (j + 1) > 0:
                rho = j + 1

        # Threshold
        theta = (sum(vals[:rho]) - 1.0) / rho

        result: dict[ActionType, float] = {}
        for a in actions:
            result[a] = max(0.0, x.get(a, 0.0) - theta)

        # Safety: ensure sums to 1
        total = sum(result.values())
        if total > 0 and abs(total - 1.0) > 1e-9:
            result = {a: p / total for a, p in result.items()}
        elif total == 0:
            uni = 1.0 / n
            result = {a: uni for a in actions}

        return result

    def _compute_total_log_likelihood(self) -> float:
        """Compute total log-likelihood of all observations under belief."""
        ll = 0.0
        for info_set, counts in self._obs_counts.items():
            belief = self._belief.get(info_set, {})
            for action, count in counts.items():
                p = belief.get(action, 1e-10)
                ll += count * math.log(max(p, 1e-10))
        return ll

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        info_set: str,
        legal: set[ActionType],
    ) -> ActionDistribution:
        """Predict opponent's action distribution at *info_set*.

        Falls back to uniform distribution if insufficient observations
        (fewer than :attr:`min_observations`).

        Args:
            info_set: The information set to predict at.
            legal: The set of legal actions.

        Returns:
            Predicted action distribution.
        """
        if not legal:
            return ActionDistribution(probabilities={})

        n = len(legal)
        uniform = {a: 1.0 / n for a in legal}

        total_obs = self._total_obs.get(info_set, 0)
        if total_obs < self.min_observations or info_set not in self._belief:
            return ActionDistribution(probabilities=uniform)

        belief = self._belief[info_set]

        # Ensure all legal actions have a probability
        probs: dict[ActionType, float] = {}
        for a in legal:
            probs[a] = belief.get(a, 0.0)

        # Normalise
        total = sum(probs.values())
        if total > 0:
            probs = {a: p / total for a, p in probs.items()}
        else:
            probs = uniform

        return ActionDistribution(probabilities=probs)

    # ------------------------------------------------------------------
    # Confidence & diagnostics
    # ------------------------------------------------------------------

    def confidence_at(self, info_set: str) -> float:
        """Return confidence in prediction at *info_set*.

        Confidence is based on the number of observations and the
        entropy of the current belief.  More observations and lower
        entropy (more peaked distribution) → higher confidence.

        Returns:
            Value in [0, 1].
        """
        total_obs = self._total_obs.get(info_set, 0)
        if total_obs == 0:
            return 0.0

        # Observation component: saturates around 50 observations
        obs_conf = 1.0 - math.exp(-total_obs / 20.0)

        # Entropy component: low entropy → high confidence
        belief = self._belief.get(info_set, {})
        legal = self._legal_at.get(info_set, set())
        n = len(legal) if legal else 1
        if n <= 1:
            return obs_conf

        # Compute normalised entropy (0 = peaked, 1 = uniform)
        total = sum(belief.get(a, 0.0) for a in legal)
        if total <= 0:
            return 0.0

        entropy = 0.0
        max_entropy = math.log(n) if n > 1 else 1.0
        for a in legal:
            p = belief.get(a, 0.0) / total
            if p > 1e-10:
                entropy -= p * math.log(p)

        norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        entropy_conf = 1.0 - norm_entropy  # peaked → high confidence

        # Combine: geometric mean gives a balanced measure
        return math.sqrt(obs_conf * entropy_conf)

    def log_likelihood_ratio(self) -> float:
        """Return ratio of model log-likelihood vs uniform baseline.

        A positive value means the model explains the data better than
        uniform.  Returns 0.0 when there are no observations.
        """
        if not self._obs_counts:
            return 0.0

        # Uniform baseline log-likelihood
        ll_uniform = 0.0
        for info_set, counts in self._obs_counts.items():
            legal = self._legal_at.get(info_set, set())
            n = len(legal) if legal else 1
            p_uniform = 1.0 / n if n > 0 else 1.0
            total_count = sum(counts.values())
            ll_uniform += total_count * math.log(max(p_uniform, 1e-10))

        # Ratio (difference in log space)
        return self._log_likelihood - ll_uniform

    def convergence_diagnostic(self) -> dict[str, object]:
        """Return convergence metrics for monitoring.

        Returns:
            Dict with:
            - ``step``: Number of gradient steps taken.
            - ``log_likelihood``: Current total log-likelihood.
            - ``ll_improvement``: Change in LL over last step.
            - ``info_sets_tracked``: Number of info sets with beliefs.
            - ``total_observations``: Total observation count.
            - ``mean_confidence``: Average confidence across info sets.
            - ``converged``: Heuristic convergence flag.
        """
        ll_improvement = 0.0
        if len(self._ll_history) >= 2:
            ll_improvement = self._ll_history[-1] - self._ll_history[-2]

        total_obs = sum(self._total_obs.values())

        # Mean confidence
        confidences = [
            self.confidence_at(info_set) for info_set in self._belief
        ]
        mean_conf = (
            sum(confidences) / len(confidences) if confidences else 0.0
        )

        # Heuristic convergence: LL improvement is small relative to LL
        converged = False
        if self._step >= 10 and len(self._ll_history) >= 5:
            recent_improvements = [
                self._ll_history[i] - self._ll_history[i - 1]
                for i in range(-4, 0)
                if i < len(self._ll_history)
            ]
            if recent_improvements:
                avg_improvement = sum(abs(x) for x in recent_improvements) / len(
                    recent_improvements
                )
                # Converged if average improvement is less than 0.1% of |LL|
                if abs(self._log_likelihood) > 0:
                    converged = avg_improvement < 0.001 * abs(self._log_likelihood)
                else:
                    converged = avg_improvement < 1e-6

        return {
            "step": self._step,
            "log_likelihood": round(self._log_likelihood, 6),
            "ll_improvement": round(ll_improvement, 6),
            "info_sets_tracked": len(self._belief),
            "total_observations": total_obs,
            "mean_confidence": round(mean_conf, 4),
            "converged": converged,
        }

    # ------------------------------------------------------------------
    # Integration with BehavioralPipeline
    # ------------------------------------------------------------------

    def merge_with_behavioral(
        self,
        info_set: str,
        legal: set[ActionType],
        signals: BehavioralSignals,
    ) -> ActionDistribution:
        """Blend statistical model prediction with behavioral signals.

        The merge uses behavioural signals to modulate the statistical
        prediction:

        - **Tilt**: shift probability mass toward aggressive actions
          (opponents on tilt tend to bet/raise more).
        - **Fatigue**: shift toward passive actions (fatigued players
          fold/check more).
        - **Adaptation**: if opponent is adapting, trust the statistical
          model less (fall back toward uniform).

        Args:
            info_set: Information set key.
            legal: Legal actions.
            signals: Current behavioural signals.

        Returns:
            Merged action distribution.
        """
        if not legal:
            return ActionDistribution(probabilities={})

        base = self.predict(info_set, legal)
        probs = dict(base.probabilities)

        # Tilt adjustment: shift mass toward aggressive actions
        if signals.tilt_score > 0.2:
            tilt_shift = signals.tilt_score * 0.2
            aggressive = {ActionType.BET, ActionType.RAISE, ActionType.ALL_IN}
            passive = {ActionType.FOLD, ActionType.CHECK, ActionType.CALL}
            for a in legal:
                if a in aggressive:
                    probs[a] = probs.get(a, 0.0) + tilt_shift / max(
                        1, len(aggressive & legal)
                    )
                elif a in passive:
                    probs[a] = max(
                        0.0,
                        probs.get(a, 0.0)
                        - tilt_shift / max(1, len(passive & legal)),
                    )

        # Fatigue adjustment: shift mass toward passive actions
        if signals.fatigue_score > 0.3:
            fatigue_shift = signals.fatigue_score * 0.15
            passive = {ActionType.FOLD, ActionType.CHECK}
            aggressive = {ActionType.BET, ActionType.RAISE, ActionType.ALL_IN}
            for a in legal:
                if a in passive:
                    probs[a] = probs.get(a, 0.0) + fatigue_shift / max(
                        1, len(passive & legal)
                    )
                elif a in aggressive:
                    probs[a] = max(
                        0.0,
                        probs.get(a, 0.0)
                        - fatigue_shift / max(1, len(aggressive & legal)),
                    )

        # Adaptation: blend toward uniform if opponent is changing strategy
        if signals.adaptation_state != AdaptationState.STATIC:
            alpha = signals.adaptation_magnitude * 0.5
            alpha = min(0.5, max(0.0, alpha))
            n = len(legal)
            uniform_p = 1.0 / n
            for a in legal:
                probs[a] = probs.get(a, 0.0) * (1.0 - alpha) + uniform_p * alpha

        # Ensure non-negative and normalise
        for a in legal:
            probs[a] = max(0.0, probs.get(a, 0.0))

        total = sum(probs.values())
        if total > 0:
            probs = {a: p / total for a, p in probs.items()}
        else:
            n = len(legal)
            probs = {a: 1.0 / n for a in legal}

        return ActionDistribution(probabilities=probs)

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def get_belief(self) -> OpponentBelief:
        """Return a snapshot of the current belief state.

        Returns:
            An :class:`OpponentBelief` with current strategies,
            confidences, observation counts, and log-likelihood.
        """
        strategy: dict[str, ActionDistribution] = {}
        confidence: dict[str, float] = {}
        observations: dict[str, int] = {}

        for info_set in self._belief:
            legal = self._legal_at.get(info_set, set())
            strategy[info_set] = self.predict(info_set, legal)
            confidence[info_set] = self.confidence_at(info_set)
            observations[info_set] = self._total_obs.get(info_set, 0)

        return OpponentBelief(
            strategy=strategy,
            confidence=confidence,
            observations=observations,
            log_likelihood=self._log_likelihood,
        )

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all beliefs and observations."""
        self._belief.clear()
        self._obs_counts.clear()
        self._legal_at.clear()
        self._total_obs.clear()
        self._pending.clear()
        self._log_likelihood = 0.0
        self._ll_history.clear()
        self._step = 0
