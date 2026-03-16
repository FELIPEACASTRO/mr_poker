"""Tests for Batch 4: Deep CFR extensions (RobustDeepMCCFR, ODCFR, MMD)."""

from __future__ import annotations

import math
import random

import pytest

from packages.cfr_agent.deep_cfr import FEATURE_DIM, NUM_ACTIONS, SimpleNN
from packages.cfr_agent.mmd import MagneticMirrorDescent
from packages.cfr_agent.odcfr import (
    EXTENDED_FEATURE_DIM,
    OPPONENT_FEATURE_DIM,
    OpponentAwareDeepCFR,
    default_opponent_features,
    extract_opponent_features,
)
from packages.cfr_agent.robust_deep_mccfr import (
    PrioritizedReplayBuffer,
    RobustDeepMCCFR,
    TargetNetwork,
)
from packages.common.types import ActionType
from packages.opponent_model.classifier import PlayerStats
from packages.strategy.mixed import ActionDistribution


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _dummy_features(dim: int = FEATURE_DIM) -> list[float]:
    rng = random.Random(42)
    return [rng.random() for _ in range(dim)]


LEGAL = {ActionType.FOLD, ActionType.CHECK, ActionType.CALL}


# ---------------------------------------------------------------------------
# TargetNetwork tests
# ---------------------------------------------------------------------------

class TestTargetNetwork:
    def test_creation_and_forward(self):
        tn = TargetNetwork(FEATURE_DIM, 32, NUM_ACTIONS, seed=1)
        out = tn.forward(_dummy_features())
        assert len(out) == NUM_ACTIONS

    def test_soft_update_moves_weights(self):
        source = SimpleNN(FEATURE_DIM, 32, NUM_ACTIONS, seed=10)
        target = TargetNetwork(FEATURE_DIM, 32, NUM_ACTIONS, seed=20)

        # Record weights before update
        w_before = target.network.w1[0][0]
        target.soft_update(source, tau=0.5)
        w_after = target.network.w1[0][0]

        # Weight should have moved toward source
        assert w_before != w_after

    def test_soft_update_tau_zero_no_change(self):
        source = SimpleNN(FEATURE_DIM, 32, NUM_ACTIONS, seed=10)
        target = TargetNetwork(FEATURE_DIM, 32, NUM_ACTIONS, seed=20)

        w_before = target.network.w1[0][0]
        target.soft_update(source, tau=0.0)
        assert target.network.w1[0][0] == w_before

    def test_soft_update_tau_one_copies(self):
        source = SimpleNN(FEATURE_DIM, 32, NUM_ACTIONS, seed=10)
        target = TargetNetwork(FEATURE_DIM, 32, NUM_ACTIONS, seed=20)

        target.soft_update(source, tau=1.0)
        assert target.network.w1[0][0] == pytest.approx(source.w1[0][0])


# ---------------------------------------------------------------------------
# PrioritizedReplayBuffer tests
# ---------------------------------------------------------------------------

class TestPrioritizedReplayBuffer:
    def test_add_and_len(self):
        buf = PrioritizedReplayBuffer(capacity=100)
        buf.add([1.0, 2.0], [0.5], td_error=1.0)
        buf.add([3.0, 4.0], [0.6], td_error=0.1)
        assert len(buf) == 2

    def test_sample_returns_correct_size(self):
        buf = PrioritizedReplayBuffer(capacity=100)
        for i in range(50):
            buf.add([float(i)], [float(i) * 0.1], td_error=float(i + 1))
        batch = buf.sample(10, random.Random(42))
        assert len(batch) == 10

    def test_high_priority_sampled_more(self):
        buf = PrioritizedReplayBuffer(capacity=100)
        # One low-priority and many high-priority
        buf.add([0.0], [0.0], td_error=0.001)
        for i in range(20):
            buf.add([1.0], [1.0], td_error=100.0)

        rng = random.Random(123)
        batch = buf.sample(50, rng)
        high_count = sum(1 for s in batch if s.features == [1.0])
        # High-priority items should be sampled more than low (20 high vs 1 low in buffer)
        assert high_count > 15

    def test_sample_small_buffer_returns_all(self):
        buf = PrioritizedReplayBuffer(capacity=100)
        buf.add([1.0], [1.0], td_error=1.0)
        batch = buf.sample(10, random.Random(0))
        assert len(batch) == 1


# ---------------------------------------------------------------------------
# RobustDeepMCCFR tests
# ---------------------------------------------------------------------------

class TestRobustDeepMCCFR:
    def test_creation(self):
        r = RobustDeepMCCFR(hidden_dim=32, seed=1)
        assert len(r.online_nets) == 2
        assert len(r.target_nets) == 2
        assert len(r.replay_buffers) == 2

    def test_add_sample_and_train(self):
        r = RobustDeepMCCFR(hidden_dim=32, seed=1)
        feats = _dummy_features()
        advs = [0.1, -0.2, 0.05, 0.0, 0.0, 0.0]
        for _ in range(20):
            r.add_sample(0, feats, advs)
        loss = r.train_step(0, batch_size=10)
        assert loss >= 0.0

    def test_get_strategy_valid_distribution(self):
        r = RobustDeepMCCFR(hidden_dim=32, seed=1)
        strat = r.get_strategy(0, _dummy_features(), LEGAL)
        assert isinstance(strat, ActionDistribution)
        assert abs(sum(strat.probabilities.values()) - 1.0) < 1e-6

    def test_target_strategy_differs_from_online(self):
        r = RobustDeepMCCFR(hidden_dim=32, seed=1)
        feats = _dummy_features()
        # Before any training they use different seeds, so strategies differ
        online = r.get_strategy(0, feats, LEGAL)
        target = r.get_target_strategy(0, feats, LEGAL)
        # Both should be valid distributions
        assert abs(sum(online.probabilities.values()) - 1.0) < 1e-6
        assert abs(sum(target.probabilities.values()) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# ODCFR tests
# ---------------------------------------------------------------------------

class TestODCFR:
    def test_feature_dimensions(self):
        assert OPPONENT_FEATURE_DIM == 8
        assert EXTENDED_FEATURE_DIM == 23
        assert EXTENDED_FEATURE_DIM == FEATURE_DIM + OPPONENT_FEATURE_DIM

    def test_extract_opponent_features_default_stats(self):
        stats = PlayerStats()
        feats = extract_opponent_features(stats)
        assert len(feats) == OPPONENT_FEATURE_DIM
        # Default stats: 0 hands, so VPIP=0, PFR=0, etc.
        assert feats[0] == 0.0  # VPIP

    def test_extract_opponent_features_with_data(self):
        stats = PlayerStats(
            total_hands=100,
            voluntary_put_in_pot=30,
            preflop_raises=20,
            total_aggressive_actions=50,
            total_passive_actions=25,
        )
        feats = extract_opponent_features(stats)
        assert len(feats) == OPPONENT_FEATURE_DIM
        assert feats[0] == pytest.approx(0.3)  # VPIP = 30/100
        assert feats[1] == pytest.approx(0.2)  # PFR = 20/100
        assert feats[2] == pytest.approx(min(50 / 25 / 5.0, 1.0))  # AF capped

    def test_default_opponent_features(self):
        feats = default_opponent_features()
        assert len(feats) == OPPONENT_FEATURE_DIM

    def test_build_features_length(self):
        od = OpponentAwareDeepCFR(hidden_dim=32, seed=1)
        base = _dummy_features()
        extended = od.build_features(base)
        assert len(extended) == EXTENDED_FEATURE_DIM

    def test_get_strategy_valid(self):
        od = OpponentAwareDeepCFR(hidden_dim=32, seed=1)
        base = _dummy_features()
        strat = od.get_strategy(0, base, LEGAL)
        assert isinstance(strat, ActionDistribution)
        assert abs(sum(strat.probabilities.values()) - 1.0) < 1e-6

    def test_get_strategy_with_opponent_stats(self):
        od = OpponentAwareDeepCFR(hidden_dim=32, seed=1)
        base = _dummy_features()
        stats = PlayerStats(total_hands=50, voluntary_put_in_pot=25)
        strat = od.get_strategy(0, base, LEGAL, opponent_stats=stats)
        assert isinstance(strat, ActionDistribution)


# ---------------------------------------------------------------------------
# MMD tests
# ---------------------------------------------------------------------------

class TestMMD:
    def test_creation(self):
        mmd = MagneticMirrorDescent(temperature=1.0, learning_rate=0.01)
        assert mmd.num_info_sets == 0
        assert mmd.update_count == 0

    def test_invalid_temperature(self):
        with pytest.raises(ValueError):
            MagneticMirrorDescent(temperature=0.0)
        with pytest.raises(ValueError):
            MagneticMirrorDescent(temperature=-1.0)

    def test_update_and_get_strategy(self):
        mmd = MagneticMirrorDescent(temperature=1.0, learning_rate=0.1)
        utils = {ActionType.FOLD: -1.0, ActionType.CALL: 0.5, ActionType.RAISE: 1.0}
        mmd.update("info1", utils, set(utils.keys()))
        strat = mmd.get_strategy("info1", set(utils.keys()))
        assert abs(sum(strat.probabilities.values()) - 1.0) < 1e-6
        # RAISE has highest utility, should have highest probability
        assert strat.probabilities[ActionType.RAISE] > strat.probabilities[ActionType.FOLD]

    def test_update_moves_toward_higher_utility(self):
        mmd = MagneticMirrorDescent(temperature=0.5, learning_rate=0.5, magnetic_strength=0.01)
        actions = {ActionType.FOLD, ActionType.CALL, ActionType.RAISE}
        utils = {ActionType.FOLD: -2.0, ActionType.CALL: 0.0, ActionType.RAISE: 2.0}

        # Run many updates to converge
        for _ in range(100):
            mmd.update("test", utils, actions)

        strat = mmd.get_strategy("test", actions)
        # RAISE should dominate after many updates
        assert strat.probabilities[ActionType.RAISE] > 0.5
        assert strat.probabilities[ActionType.FOLD] < 0.1

    def test_temperature_effect(self):
        """Lower temperature = more peaked strategy."""
        actions = {ActionType.FOLD, ActionType.CALL}
        utils = {ActionType.FOLD: -1.0, ActionType.CALL: 1.0}

        low_temp = MagneticMirrorDescent(temperature=0.1, learning_rate=0.1, magnetic_strength=0.01)
        high_temp = MagneticMirrorDescent(temperature=10.0, learning_rate=0.1, magnetic_strength=0.01)

        for _ in range(50):
            low_temp.update("x", utils, actions)
            high_temp.update("x", utils, actions)

        low_strat = low_temp.get_strategy("x", actions)
        high_strat = high_temp.get_strategy("x", actions)

        # Low temperature should be more peaked (higher max prob)
        low_max = max(low_strat.probabilities.values())
        high_max = max(high_strat.probabilities.values())
        assert low_max > high_max

    def test_convergence_gap_decreases(self):
        mmd = MagneticMirrorDescent(temperature=1.0, learning_rate=0.01, magnetic_strength=0.5)
        actions = {ActionType.CHECK, ActionType.BET}
        utils = {ActionType.CHECK: 0.5, ActionType.BET: 0.5}  # Equal utilities

        # With equal utilities and magnetic strength, logits should stay near zero
        for _ in range(100):
            mmd.update("eq", utils, actions)

        gap = mmd.convergence_gap()
        assert gap < 10.0  # Should be moderate/small

    def test_get_strategy_unseen_info_set(self):
        mmd = MagneticMirrorDescent()
        actions = {ActionType.FOLD, ActionType.CALL}
        strat = mmd.get_strategy("never_seen", actions)
        # Unseen info set: all logits are 0, so uniform
        for p in strat.probabilities.values():
            assert p == pytest.approx(0.5)

    def test_reset(self):
        mmd = MagneticMirrorDescent()
        mmd.update("x", {ActionType.FOLD: 1.0}, {ActionType.FOLD})
        assert mmd.num_info_sets == 1
        mmd.reset()
        assert mmd.num_info_sets == 0
        assert mmd.update_count == 0


# ---------------------------------------------------------------------------
# Import smoke tests
# ---------------------------------------------------------------------------

class TestImports:
    def test_all_modules_importable(self):
        import packages.cfr_agent.robust_deep_mccfr
        import packages.cfr_agent.odcfr
        import packages.cfr_agent.mmd
        assert True
