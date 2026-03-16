"""DDM Timing — Drift Diffusion Model for decision-time preference inference.

Implements a Drift Diffusion Model (DDM) to formally estimate opponent
preferences and certainty from their decision response times.

Scientific basis: "Estimating Preferences Using Response Time Data"
(arxiv.org/abs/2507.20403) — DDM achieves 1/n convergence rate for
preference estimation using response times, vs 1/sqrt(n) for choice-only.

Key insight: Fast decisions indicate high certainty about the chosen action.
Slow decisions indicate difficulty, which reveals information about the
opponent's hand strength and strategic considerations.

The DDM models decision-making as a noisy accumulation process:
    dx = drift * dt + noise * dW

Where:
- drift = preference strength (how much they favor one action)
- noise = decision difficulty
- boundary = confidence threshold to act

Components:
- DDMEstimator: fits DDM parameters from observed (time, action) pairs
- PreferenceEstimate: structured output with certainty and strength

Reference: arxiv.org/abs/2507.20403
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class DDMParameters:
    """Fitted DDM parameters for a decision context.

    drift: rate of evidence accumulation (higher = clearer preference)
    boundary: decision threshold (higher = more deliberate player)
    noise: noise level (reflects decision difficulty)
    non_decision_time: motor/perceptual baseline time (ms)
    """

    drift: float = 1.0
    boundary: float = 1.5
    noise: float = 1.0
    non_decision_time: float = 500.0  # ms


@dataclass
class PreferenceEstimate:
    """Estimated preference from DDM analysis.

    certainty: 0-1 confidence in the chosen action (from drift rate)
    preference_strength: how strongly they favor the action (from drift magnitude)
    is_conflicted: high decision time suggests internal conflict
    estimated_ev_gap: estimated EV difference between best and second-best action
    """

    certainty: float = 0.5
    preference_strength: float = 0.0
    is_conflicted: bool = False
    estimated_ev_gap: float = 0.0
    decision_time_ms: float = 0.0


@dataclass
class DDMObservation:
    """A single observed decision with timing."""

    context: str  # situation key (e.g., "flop_cbet_IP")
    action: str   # chosen action value
    decision_time_ms: float
    was_aggressive: bool = False


class DDMEstimator:
    """Fits DDM parameters from observed decision times.

    Maintains per-context statistics and estimates DDM parameters
    using the EZ-diffusion method (Wagenmakers et al., 2007) which
    provides closed-form estimates from mean RT, RT variance, and accuracy.

    Usage::

        estimator = DDMEstimator()
        estimator.observe("flop_cbet", "bet", 1200)
        estimator.observe("flop_cbet", "check", 3500)
        estimator.observe("flop_cbet", "bet", 800)

        params = estimator.fit("flop_cbet")
        pref = estimator.estimate_preference("flop_cbet", 600, "bet")
    """

    def __init__(self, baseline_rt_ms: float = 500.0) -> None:
        self.baseline_rt = baseline_rt_ms
        # context -> list of (action, time_ms)
        self._observations: dict[str, list[tuple[str, float]]] = defaultdict(list)

    def observe(self, context: str, action: str, decision_time_ms: float) -> None:
        """Record a decision observation."""
        self._observations[context].append((action, decision_time_ms))

    def fit(self, context: str) -> DDMParameters:
        """Fit DDM parameters for a given context using EZ-diffusion.

        EZ-diffusion (Wagenmakers 2007) provides closed-form estimates:
        - drift from mean RT and accuracy
        - boundary from RT variance
        - non-decision time from minimum RT
        """
        obs = self._observations.get(context, [])
        if len(obs) < 3:
            return DDMParameters()

        times = [t for _, t in obs]
        mean_rt = sum(times) / len(times)
        var_rt = sum((t - mean_rt) ** 2 for t in times) / len(times)

        # Accuracy = proportion of most common action
        action_counts: dict[str, int] = defaultdict(int)
        for action, _ in obs:
            action_counts[action] += 1
        total = len(obs)
        accuracy = max(action_counts.values()) / total

        # EZ-diffusion equations (simplified)
        # Clamp accuracy to avoid log(0) or log(negative)
        accuracy = max(0.51, min(0.99, accuracy))

        # Logit of accuracy
        logit_acc = math.log(accuracy / (1 - accuracy))

        # Drift rate estimate
        # v = sign(acc - 0.5) * s * logit(acc) / sqrt(VRT)
        s = 1.0  # scaling parameter
        drift = s * logit_acc / max(math.sqrt(var_rt / 1e6), 0.01)

        # Boundary estimate
        # a = s^2 * logit(acc) / v
        boundary = s * s * logit_acc / max(abs(drift), 0.01)

        # Non-decision time: minimum observed RT (approximate)
        non_decision = min(times)

        return DDMParameters(
            drift=drift,
            boundary=abs(boundary),
            noise=s,
            non_decision_time=non_decision,
        )

    def estimate_preference(
        self,
        context: str,
        decision_time_ms: float,
        chosen_action: str,
    ) -> PreferenceEstimate:
        """Estimate preference/certainty from a single decision time.

        Uses fitted DDM parameters for the context to infer how
        certain the opponent was about their choice.
        """
        params = self.fit(context)
        obs = self._observations.get(context, [])

        if len(obs) < 3:
            return PreferenceEstimate(decision_time_ms=decision_time_ms)

        times = [t for _, t in obs]
        mean_rt = sum(times) / len(times)
        std_rt = math.sqrt(sum((t - mean_rt) ** 2 for t in times) / len(times))

        # How fast relative to their baseline?
        # z-score: negative = faster than average = more certain
        if std_rt > 0:
            z_score = (decision_time_ms - mean_rt) / std_rt
        else:
            # Zero variance: use relative deviation from mean as proxy
            if mean_rt > 0:
                z_score = (decision_time_ms - mean_rt) / (mean_rt * 0.1)
            else:
                z_score = 0.0

        # Certainty: sigmoid transform of negative z-score
        # Fast decision → high certainty, slow → low certainty
        certainty = 1.0 / (1.0 + math.exp(z_score))

        # Preference strength from drift rate
        preference_strength = abs(params.drift)

        # Conflict detection: very slow decisions suggest internal conflict
        is_conflicted = z_score > 1.5

        # Estimated EV gap: higher drift = bigger difference between options
        # Approximate: drift * boundary gives evidence at decision point
        ev_gap = abs(params.drift * params.boundary)

        return PreferenceEstimate(
            certainty=certainty,
            preference_strength=min(1.0, preference_strength),
            is_conflicted=is_conflicted,
            estimated_ev_gap=ev_gap,
            decision_time_ms=decision_time_ms,
        )

    @property
    def contexts_observed(self) -> int:
        return len(self._observations)

    @property
    def total_observations(self) -> int:
        return sum(len(v) for v in self._observations.values())
