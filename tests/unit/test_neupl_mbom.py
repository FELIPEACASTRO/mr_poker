"""Tests for NeuPL (Neural Population Learning) and MBOM (Model-Based Opponent Modeling)."""

import math
import pytest

from packages.cfr_agent.neupl import (
    NeuPLNetwork,
    NeuPLPopulation,
    PopulationPolicy,
)
from packages.common.types import ActionType
from packages.opponent_model.mbom import (
    BeliefState,
    OpponentModel,
    RecursiveModeler,
)
from packages.strategy.mixed import ActionDistribution


# ---------------------------------------------------------------------------
# NeuPL tests
# ---------------------------------------------------------------------------


class TestNeuPLNetwork:
    def setup_method(self):
        self.net = NeuPLNetwork(feature_dim=10, embedding_dim=16, hidden_dim=24, seed=42)

    def test_forward_returns_correct_shape(self):
        features = [0.5] * 10
        embedding = [0.1] * 16
        logits = self.net.forward(features, embedding)
        assert len(logits) == 6  # 6 action types

    def test_get_strategy_valid_distribution(self):
        features = [0.5] * 10
        embedding = [0.1] * 16
        legal = {ActionType.FOLD, ActionType.CALL, ActionType.RAISE}

        strategy = self.net.get_strategy(features, embedding, legal)

        assert set(strategy.probabilities.keys()) == legal
        total = sum(strategy.probabilities.values())
        assert abs(total - 1.0) < 1e-6
        for p in strategy.probabilities.values():
            assert p >= 0

    def test_different_embeddings_different_strategies(self):
        """Different opponent embeddings should produce different strategies."""
        features = [0.5] * 10
        legal = {ActionType.FOLD, ActionType.CALL, ActionType.RAISE}

        emb_nit = [1.0, 0.0, 0.0] + [0.0] * 13  # "nit-like"
        emb_lag = [0.0, 1.0, 1.0] + [0.0] * 13  # "lag-like"

        strat_nit = self.net.get_strategy(features, emb_nit, legal)
        strat_lag = self.net.get_strategy(features, emb_lag, legal)

        # Strategies should differ for different embeddings
        diff = sum(
            abs(strat_nit.probabilities.get(a, 0) - strat_lag.probabilities.get(a, 0))
            for a in legal
        )
        assert diff > 0.001, "Different embeddings should produce different strategies"

    def test_train_step_decreases_loss(self):
        features = [0.5] * 10
        embedding = [0.3] * 16
        target = {ActionType.RAISE: 0.8, ActionType.CALL: 0.15, ActionType.FOLD: 0.05}

        losses = []
        for _ in range(50):
            loss = self.net.train_step(features, embedding, target, lr=0.01)
            losses.append(loss)

        # Loss should decrease over training
        assert losses[-1] < losses[0], f"Loss should decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"

    def test_train_step_increments_counter(self):
        assert self.net.train_steps == 0
        self.net.train_step([0.5] * 10, [0.1] * 16, {ActionType.CHECK: 1.0})
        assert self.net.train_steps == 1

    def test_legal_action_masking(self):
        features = [0.5] * 10
        embedding = [0.1] * 16
        legal = {ActionType.CHECK, ActionType.BET}  # Only 2 actions legal

        strategy = self.net.get_strategy(features, embedding, legal)
        assert set(strategy.probabilities.keys()) == legal
        # No probability on illegal actions
        assert ActionType.FOLD not in strategy.probabilities

    def test_short_feature_vector_padded(self):
        """Feature vectors shorter than expected should be zero-padded."""
        features = [0.5] * 5  # shorter than feature_dim=10
        embedding = [0.1] * 16
        legal = {ActionType.CHECK, ActionType.BET}

        strategy = self.net.get_strategy(features, embedding, legal)
        assert abs(sum(strategy.probabilities.values()) - 1.0) < 1e-6


class TestNeuPLPopulation:
    def setup_method(self):
        self.pop = NeuPLPopulation(feature_dim=10, embedding_dim=16, seed=42)

    def test_add_archetype(self):
        emb = [0.5] * 16
        self.pop.add_archetype("nit", emb, "Tight passive player")
        assert "nit" in self.pop.archetype_names

    def test_get_archetype_strategy(self):
        self.pop.add_archetype("nit", [0.1] * 16)
        self.pop.add_archetype("lag", [0.9] * 16)

        features = [0.5] * 10
        legal = {ActionType.FOLD, ActionType.CALL, ActionType.RAISE}

        strat = self.pop.get_archetype_strategy(features, "nit", legal)
        assert abs(sum(strat.probabilities.values()) - 1.0) < 1e-6

    def test_unknown_archetype_uses_zero_embedding(self):
        features = [0.5] * 10
        legal = {ActionType.CHECK, ActionType.BET}

        strat = self.pop.get_archetype_strategy(features, "unknown_type", legal)
        assert abs(sum(strat.probabilities.values()) - 1.0) < 1e-6

    def test_get_counter_strategy(self):
        features = [0.5] * 10
        embedding = [0.3] * 16
        legal = {ActionType.FOLD, ActionType.CALL, ActionType.RAISE}

        strat = self.pop.get_counter_strategy(features, embedding, legal)
        assert set(strat.probabilities.keys()) == legal

    def test_train_against_archetype(self):
        self.pop.add_archetype("fish", [0.8, 0.2] + [0.0] * 14)
        features = [0.5] * 10
        target = {ActionType.BET: 0.7, ActionType.CHECK: 0.3}

        loss = self.pop.train_against_archetype(features, "fish", target)
        assert loss > 0

    def test_train_unknown_archetype_returns_zero(self):
        loss = self.pop.train_against_archetype([0.5] * 10, "nonexistent", {ActionType.CHECK: 1.0})
        assert loss == 0.0


# ---------------------------------------------------------------------------
# MBOM tests
# ---------------------------------------------------------------------------


class TestBeliefState:
    def test_observe_and_predict(self):
        beliefs = BeliefState()
        legal = {ActionType.FOLD, ActionType.CALL, ActionType.RAISE}

        # Observe actions
        for _ in range(10):
            beliefs.observe("flop_cbet", ActionType.FOLD)
        for _ in range(5):
            beliefs.observe("flop_cbet", ActionType.CALL)
        beliefs.observe("flop_cbet", ActionType.RAISE)

        pred = beliefs.predict("flop_cbet", legal)
        # FOLD should have highest probability
        assert pred.probabilities[ActionType.FOLD] > pred.probabilities[ActionType.CALL]
        assert pred.probabilities[ActionType.CALL] > pred.probabilities[ActionType.RAISE]

    def test_bayesian_smoothing(self):
        """With no observations, prediction should be near-uniform."""
        beliefs = BeliefState(prior_strength=2.0)
        legal = {ActionType.FOLD, ActionType.CALL, ActionType.RAISE}

        pred = beliefs.predict("unseen_situation", legal)
        for p in pred.probabilities.values():
            assert abs(p - 1.0 / 3) < 0.01

    def test_confidence_increases(self):
        beliefs = BeliefState()
        assert beliefs.confidence("sit1") < 0.01  # No data

        for _ in range(30):
            beliefs.observe("sit1", ActionType.FOLD)
        assert beliefs.confidence("sit1") > 0.7

    def test_total_observations(self):
        beliefs = BeliefState()
        beliefs.observe("s1", ActionType.FOLD)
        beliefs.observe("s1", ActionType.CALL)
        beliefs.observe("s2", ActionType.RAISE)
        assert beliefs.total_observations == 3


class TestOpponentModel:
    def test_observe_and_predict(self):
        model = OpponentModel()
        legal = {ActionType.FOLD, ActionType.CALL}

        for _ in range(20):
            model.observe("preflop_3bet", ActionType.FOLD)
        for _ in range(5):
            model.observe("preflop_3bet", ActionType.CALL)

        pred = model.predict("preflop_3bet", legal)
        assert pred.probabilities[ActionType.FOLD] > pred.probabilities[ActionType.CALL]

    def test_recency_weighting(self):
        """Recent observations should slightly influence predictions."""
        model = OpponentModel(recency_weight=0.3)
        legal = {ActionType.FOLD, ActionType.CALL}

        # Long history of folding
        for _ in range(50):
            model.observe("sit1", ActionType.FOLD)

        # Recent switch to calling
        for _ in range(10):
            model.observe("sit1", ActionType.CALL)

        pred = model.predict("sit1", legal)
        # Should still favor fold (long history) but call should be non-trivial
        assert pred.probabilities[ActionType.CALL] > 0.1

    def test_total_observations(self):
        model = OpponentModel()
        model.observe("s1", ActionType.FOLD)
        model.observe("s2", ActionType.CALL)
        assert model.total_observations == 2


class TestRecursiveModeler:
    def setup_method(self):
        self.modeler = RecursiveModeler(max_level=2, seed=42)

    def test_observe_opponent(self):
        self.modeler.observe_opponent("flop_cbet", ActionType.FOLD)
        assert self.modeler._models[0].total_observations == 1

    def test_observe_our_action(self):
        self.modeler.observe_our_action("preflop", ActionType.RAISE)
        assert self.modeler._models[1].total_observations == 1

    def test_predict_level_0(self):
        legal = {ActionType.FOLD, ActionType.CALL, ActionType.RAISE}

        for _ in range(15):
            self.modeler.observe_opponent("turn_probe", ActionType.FOLD)
        for _ in range(5):
            self.modeler.observe_opponent("turn_probe", ActionType.CALL)

        pred = self.modeler.predict("turn_probe", legal, level=0)
        assert pred.probabilities[ActionType.FOLD] > pred.probabilities[ActionType.CALL]

    def test_predict_auto_level(self):
        """With few observations, auto-level should be 0."""
        legal = {ActionType.FOLD, ActionType.CALL}
        self.modeler.observe_opponent("sit", ActionType.FOLD)

        pred = self.modeler.predict("sit", legal)
        assert abs(sum(pred.probabilities.values()) - 1.0) < 1e-6

    def test_exploitation_weight(self):
        # No observations → low weight
        assert self.modeler.exploitation_weight("unknown") < 0.1

        # Many observations → higher weight
        for _ in range(50):
            self.modeler.observe_opponent("well_known", ActionType.FOLD)
        assert self.modeler.exploitation_weight("well_known") > 0.5

    def test_estimated_thinking_level(self):
        # Few observations → level 0
        assert self.modeler.estimated_thinking_level() == 0

        # Add level-0 observations
        for _ in range(25):
            self.modeler.observe_opponent("s1", ActionType.FOLD)

        # Still level 0 (no level-1 data)
        assert self.modeler.estimated_thinking_level() == 0

        # Add level-1 observations (our actions seen by opponent)
        for _ in range(25):
            self.modeler.observe_our_action("s1", ActionType.RAISE)

        # Now should detect level 1
        assert self.modeler.estimated_thinking_level() == 1

    def test_reset(self):
        for _ in range(10):
            self.modeler.observe_opponent("s1", ActionType.FOLD)
        assert self.modeler._models[0].total_observations == 10

        self.modeler.reset()
        assert self.modeler._models[0].total_observations == 0

    def test_max_level_respected(self):
        """Prediction at level > max_level should use max_level."""
        legal = {ActionType.FOLD, ActionType.CALL}
        self.modeler.observe_opponent("s1", ActionType.FOLD)

        pred = self.modeler.predict("s1", legal, level=99)
        assert abs(sum(pred.probabilities.values()) - 1.0) < 1e-6
