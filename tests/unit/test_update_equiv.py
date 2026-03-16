"""Tests for Update-Equivalence decision-time planning framework."""

import math
import pytest

from packages.cfr_agent.update_equiv import (
    DecisionTimePlanner,
    GradientEstimate,
    GradientEstimator,
    MirrorDescentRefiner,
    RolloutResult,
    _entropy_regularized_projection,
    _kl_divergence,
    _softmax,
)
from packages.common.types import ActionType
from packages.strategy.mixed import ActionDistribution


# ---------------------------------------------------------------------------
# Helper utilities tests
# ---------------------------------------------------------------------------


class TestSoftmax:
    def test_uniform_input(self):
        vals = {ActionType.FOLD: 0.0, ActionType.CALL: 0.0, ActionType.RAISE: 0.0}
        result = _softmax(vals)
        for v in result.values():
            assert abs(v - 1.0 / 3) < 1e-6

    def test_peaked_input(self):
        vals = {ActionType.FOLD: -10.0, ActionType.CALL: 0.0, ActionType.RAISE: 10.0}
        result = _softmax(vals)
        assert result[ActionType.RAISE] > 0.99
        assert result[ActionType.FOLD] < 0.001

    def test_empty(self):
        assert _softmax({}) == {}

    def test_sums_to_one(self):
        vals = {ActionType.FOLD: 1.5, ActionType.CALL: -0.3, ActionType.BET: 2.1}
        result = _softmax(vals)
        assert abs(sum(result.values()) - 1.0) < 1e-6


class TestKLDivergence:
    def test_identical_distributions(self):
        p = {ActionType.FOLD: 0.5, ActionType.CALL: 0.5}
        assert _kl_divergence(p, p) < 1e-10

    def test_different_distributions(self):
        p = {ActionType.FOLD: 0.9, ActionType.CALL: 0.1}
        q = {ActionType.FOLD: 0.5, ActionType.CALL: 0.5}
        kl = _kl_divergence(p, q)
        assert kl > 0

    def test_non_negative(self):
        p = {ActionType.FOLD: 0.3, ActionType.CALL: 0.7}
        q = {ActionType.FOLD: 0.6, ActionType.CALL: 0.4}
        assert _kl_divergence(p, q) >= 0


class TestEntropyRegularizedProjection:
    def test_high_temperature_uniform(self):
        logits = {ActionType.FOLD: 1.0, ActionType.CALL: 1.0, ActionType.RAISE: 1.0}
        result = _entropy_regularized_projection(logits, tau=100.0)
        for v in result.values():
            assert abs(v - 1.0 / 3) < 0.01

    def test_zero_temperature_greedy(self):
        logits = {ActionType.FOLD: 1.0, ActionType.CALL: 5.0, ActionType.RAISE: 3.0}
        result = _entropy_regularized_projection(logits, tau=0.0)
        assert result[ActionType.CALL] == 1.0

    def test_low_temperature_peaked(self):
        logits = {ActionType.FOLD: 1.0, ActionType.CALL: 5.0, ActionType.RAISE: 3.0}
        result = _entropy_regularized_projection(logits, tau=0.1)
        assert result[ActionType.CALL] > 0.9


# ---------------------------------------------------------------------------
# GradientEstimator tests
# ---------------------------------------------------------------------------


class TestGradientEstimator:
    def setup_method(self):
        self.estimator = GradientEstimator(seed=42)

    def test_from_rollouts_basic(self):
        strategy = ActionDistribution(
            probabilities={ActionType.FOLD: 0.3, ActionType.CALL: 0.5, ActionType.RAISE: 0.2}
        )
        rollouts = [
            RolloutResult(ActionType.FOLD, -2.0),
            RolloutResult(ActionType.CALL, 0.5),
            RolloutResult(ActionType.CALL, 1.0),
            RolloutResult(ActionType.RAISE, 3.0),
        ]
        grad = self.estimator.estimate_from_rollouts(rollouts, strategy)
        assert len(grad.action_values) == 3
        assert grad.num_samples == 4
        assert grad.action_values[ActionType.FOLD] == -2.0
        assert abs(grad.action_values[ActionType.CALL] - 0.75) < 1e-6

    def test_from_rollouts_empty(self):
        strategy = ActionDistribution(probabilities={ActionType.CHECK: 1.0})
        grad = self.estimator.estimate_from_rollouts([], strategy)
        assert grad.num_samples == 0

    def test_from_action_values(self):
        values = {ActionType.FOLD: -1.0, ActionType.CALL: 0.5, ActionType.RAISE: 2.0}
        grad = self.estimator.estimate_from_action_values(values)
        assert grad.action_values == values
        assert grad.num_samples == 3
        assert abs(grad.baseline - 0.5) < 1e-6  # mean of -1, 0.5, 2.0

    def test_from_action_values_empty(self):
        grad = self.estimator.estimate_from_action_values({})
        assert grad.num_samples == 0


# ---------------------------------------------------------------------------
# MirrorDescentRefiner tests
# ---------------------------------------------------------------------------


class TestMirrorDescentRefiner:
    def setup_method(self):
        self.refiner = MirrorDescentRefiner(
            learning_rate=0.5,
            temperature=1.0,
            max_iterations=20,
        )

    def test_refine_toward_best_action(self):
        """Refinement should increase probability of high-value actions."""
        blueprint = ActionDistribution(
            probabilities={ActionType.FOLD: 0.33, ActionType.CALL: 0.33, ActionType.RAISE: 0.34}
        )
        gradient = GradientEstimate(
            action_values={ActionType.FOLD: -2.0, ActionType.CALL: 0.0, ActionType.RAISE: 3.0},
            num_samples=10,
            baseline=1.0 / 3,
        )
        legal = {ActionType.FOLD, ActionType.CALL, ActionType.RAISE}

        refined = self.refiner.refine(blueprint, gradient, legal)

        assert refined.probabilities[ActionType.RAISE] > blueprint.probabilities[ActionType.RAISE]
        assert refined.probabilities[ActionType.FOLD] < blueprint.probabilities[ActionType.FOLD]

    def test_refine_sums_to_one(self):
        """Refined strategy must be a valid probability distribution."""
        blueprint = ActionDistribution(
            probabilities={ActionType.CHECK: 0.6, ActionType.BET: 0.4}
        )
        gradient = GradientEstimate(
            action_values={ActionType.CHECK: -1.0, ActionType.BET: 2.0},
            num_samples=5,
            baseline=0.5,
        )
        refined = self.refiner.refine(blueprint, gradient)
        total = sum(refined.probabilities.values())
        assert abs(total - 1.0) < 1e-6

    def test_no_gradient_returns_blueprint(self):
        """With empty gradient, return blueprint unchanged."""
        blueprint = ActionDistribution(
            probabilities={ActionType.CHECK: 0.5, ActionType.BET: 0.5}
        )
        empty_grad = GradientEstimate()
        refined = self.refiner.refine(blueprint, empty_grad)
        assert refined.probabilities == blueprint.probabilities

    def test_convergence(self):
        """With strong gradient, should converge quickly."""
        refiner = MirrorDescentRefiner(
            learning_rate=1.0,
            temperature=0.5,
            max_iterations=50,
            convergence_threshold=1e-6,
        )
        blueprint = ActionDistribution(
            probabilities={ActionType.FOLD: 0.5, ActionType.RAISE: 0.5}
        )
        gradient = GradientEstimate(
            action_values={ActionType.FOLD: -5.0, ActionType.RAISE: 5.0},
            num_samples=100,
            baseline=0.0,
        )
        refined = refiner.refine(blueprint, gradient)
        # Should be heavily peaked on RAISE
        assert refined.probabilities[ActionType.RAISE] > 0.95

    def test_temperature_effect(self):
        """Lower temperature should produce more peaked distributions."""
        blueprint = ActionDistribution(
            probabilities={ActionType.FOLD: 0.5, ActionType.RAISE: 0.5}
        )
        gradient = GradientEstimate(
            action_values={ActionType.FOLD: -1.0, ActionType.RAISE: 1.0},
            num_samples=10,
            baseline=0.0,
        )

        hot = MirrorDescentRefiner(learning_rate=0.5, temperature=2.0, max_iterations=20)
        cold = MirrorDescentRefiner(learning_rate=0.5, temperature=0.2, max_iterations=20)

        refined_hot = hot.refine(blueprint, gradient)
        refined_cold = cold.refine(blueprint, gradient)

        # Cold should be more peaked than hot
        assert refined_cold.probabilities[ActionType.RAISE] >= refined_hot.probabilities[ActionType.RAISE]

    def test_legal_action_filtering(self):
        """Only legal actions should appear in output."""
        blueprint = ActionDistribution(
            probabilities={
                ActionType.FOLD: 0.2, ActionType.CHECK: 0.3,
                ActionType.CALL: 0.2, ActionType.RAISE: 0.3,
            }
        )
        gradient = GradientEstimate(
            action_values={ActionType.CHECK: 1.0, ActionType.RAISE: 2.0},
            num_samples=5,
            baseline=1.5,
        )
        legal = {ActionType.CHECK, ActionType.RAISE}

        refined = self.refiner.refine(blueprint, gradient, legal)
        assert set(refined.probabilities.keys()) == legal


# ---------------------------------------------------------------------------
# DecisionTimePlanner tests
# ---------------------------------------------------------------------------


class TestDecisionTimePlanner:
    def setup_method(self):
        self.planner = DecisionTimePlanner(
            learning_rate=0.5,
            temperature=0.5,
            max_iterations=15,
            seed=42,
        )

    def test_plan_basic(self):
        """Plan should refine blueprint toward high-value actions."""
        blueprint = ActionDistribution(
            probabilities={ActionType.CHECK: 0.5, ActionType.BET: 0.5}
        )
        legal = {ActionType.CHECK, ActionType.BET}
        values = {ActionType.CHECK: -1.0, ActionType.BET: 3.0}

        result = self.planner.plan(blueprint, legal, values)

        assert result.probabilities[ActionType.BET] > 0.5
        assert abs(sum(result.probabilities.values()) - 1.0) < 1e-6

    def test_plan_from_rollouts(self):
        """Plan from rollouts should produce valid refined strategy."""
        blueprint = ActionDistribution(
            probabilities={ActionType.FOLD: 0.33, ActionType.CALL: 0.33, ActionType.RAISE: 0.34}
        )
        legal = {ActionType.FOLD, ActionType.CALL, ActionType.RAISE}
        rollouts = [
            RolloutResult(ActionType.RAISE, 5.0),
            RolloutResult(ActionType.RAISE, 3.0),
            RolloutResult(ActionType.CALL, 0.5),
            RolloutResult(ActionType.FOLD, -2.0),
        ]

        result = self.planner.plan_from_rollouts(blueprint, legal, rollouts)

        assert result.probabilities[ActionType.RAISE] > result.probabilities[ActionType.FOLD]
        assert abs(sum(result.probabilities.values()) - 1.0) < 1e-6

    def test_refinement_count(self):
        """Refinement counter should increment."""
        blueprint = ActionDistribution(
            probabilities={ActionType.CHECK: 1.0}
        )
        legal = {ActionType.CHECK}
        values = {ActionType.CHECK: 0.0}

        assert self.planner.refinement_count == 0
        self.planner.plan(blueprint, legal, values)
        assert self.planner.refinement_count == 1
        self.planner.plan(blueprint, legal, values)
        assert self.planner.refinement_count == 2

    def test_plan_preserves_legal_actions(self):
        """Output should only contain legal actions."""
        blueprint = ActionDistribution(
            probabilities={ActionType.FOLD: 0.5, ActionType.CALL: 0.3, ActionType.RAISE: 0.2}
        )
        legal = {ActionType.CALL, ActionType.RAISE}
        values = {ActionType.CALL: 1.0, ActionType.RAISE: 2.0}

        result = self.planner.plan(blueprint, legal, values)
        assert set(result.probabilities.keys()) == legal

    def test_greedy_low_temperature(self):
        """With very low temperature, should be nearly greedy."""
        planner = DecisionTimePlanner(
            learning_rate=1.0,
            temperature=0.01,
            max_iterations=30,
        )
        blueprint = ActionDistribution(
            probabilities={ActionType.FOLD: 0.5, ActionType.RAISE: 0.5}
        )
        legal = {ActionType.FOLD, ActionType.RAISE}
        values = {ActionType.FOLD: -3.0, ActionType.RAISE: 5.0}

        result = planner.plan(blueprint, legal, values)
        assert result.probabilities[ActionType.RAISE] > 0.99

    def test_identical_values_keeps_blueprint(self):
        """When all actions have equal value, strategy shouldn't change much."""
        blueprint = ActionDistribution(
            probabilities={ActionType.CHECK: 0.6, ActionType.BET: 0.4}
        )
        legal = {ActionType.CHECK, ActionType.BET}
        values = {ActionType.CHECK: 1.0, ActionType.BET: 1.0}

        result = self.planner.plan(blueprint, legal, values)
        # With equal values, advantages are 0, so strategy should be close to original
        diff = abs(result.probabilities[ActionType.CHECK] - blueprint.probabilities[ActionType.CHECK])
        assert diff < 0.15  # Allow some drift from softmax
