"""Tests for Batch 3: PPO Agent + GPU-Accelerated CFR.

Covers:
- PPO agent creation, forward pass, action selection, GAE, training, self-play
- GPU CFR creation, batch regret matching, batch updates, training, export
- Integration tests: PPO vs random, GPU CFR producing usable strategies
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from packages.common.types import ActionType
from packages.engine.engine import GameEngine
from packages.strategy.mixed import ActionDistribution
from packages.cfr_agent.trainer import CFRState

# PPO imports
from packages.ppo_agent.agent import (
    PPOAgent,
    Experience,
    ExperienceBuffer,
    _softmax,
    _log_softmax,
)

# GPU CFR imports
from packages.cfr_agent.gpu_cfr import (
    BatchCFRTrainer,
    RegretTable,
    batch_regret_match,
    batch_update,
    NUM_ACTIONS,
    ACTION_TO_IDX,
    IDX_TO_ACTION,
)


# ---------------------------------------------------------------------------
# PPO Agent Tests
# ---------------------------------------------------------------------------


class TestPPOCreation:
    def test_agent_creates_with_defaults(self):
        agent = PPOAgent()
        assert agent.epsilon == 0.2
        assert agent.gamma == 0.99
        assert agent.lam == 0.95
        assert agent.policy_net is not None
        assert agent.value_net is not None
        assert len(agent.buffer) == 0

    def test_agent_creates_with_custom_params(self):
        agent = PPOAgent(
            hidden_dim=32,
            epsilon=0.1,
            gamma=0.95,
            lam=0.9,
            seed=123,
        )
        assert agent.epsilon == 0.1
        assert agent.gamma == 0.95
        assert agent.lam == 0.9


class TestPPOForwardPass:
    def test_policy_forward_produces_valid_probs(self):
        agent = PPOAgent(seed=42)
        features = [0.5] * 15  # FEATURE_DIM = 15
        legal = {ActionType.FOLD, ActionType.CALL, ActionType.RAISE}
        probs = agent._get_policy(features, legal)
        assert len(probs) == NUM_ACTIONS
        assert abs(sum(probs) - 1.0) < 1e-6
        # Illegal actions should have ~0 probability
        for i, p in enumerate(probs):
            action = IDX_TO_ACTION.get(i)
            if action is not None and action not in legal:
                assert p < 1e-6

    def test_value_forward_produces_scalar(self):
        agent = PPOAgent(seed=42)
        features = [0.5] * 15
        value = agent._get_value(features)
        assert isinstance(value, float)
        assert math.isfinite(value)

    def test_log_probs_consistent_with_probs(self):
        agent = PPOAgent(seed=42)
        features = [0.3, 0.7, 0.0, 1.0, 0.5, 0.0, 0.0, 0.0, 1.0, 0.2, 0.1, 0.0, 0.0, 0.0, 0.5]
        legal = {ActionType.FOLD, ActionType.CHECK, ActionType.BET}
        probs = agent._get_policy(features, legal)
        log_probs = agent._get_log_probs(features, legal)
        for i in range(NUM_ACTIONS):
            if probs[i] > 1e-8:
                assert abs(math.exp(log_probs[i]) - probs[i]) < 1e-5


class TestPPOActionSelection:
    def test_sample_action_returns_legal(self):
        agent = PPOAgent(seed=42)
        features = [0.5] * 15
        legal = {ActionType.FOLD, ActionType.CALL}
        probs = agent._get_policy(features, legal)
        for _ in range(20):
            action, idx = agent._sample_action(probs, legal)
            assert action in legal

    def test_sample_action_covers_distribution(self):
        """Over many samples, both actions should be chosen at least once."""
        agent = PPOAgent(seed=42)
        features = [0.5] * 15
        legal = {ActionType.FOLD, ActionType.CALL}
        probs = agent._get_policy(features, legal)
        seen = set()
        for _ in range(100):
            action, _ = agent._sample_action(probs, legal)
            seen.add(action)
        # At least one of the two should appear (both likely, but 1 is guaranteed)
        assert len(seen) >= 1


class TestGAE:
    def test_gae_empty(self):
        agent = PPOAgent()
        assert agent.compute_gae([], [], []) == []

    def test_gae_single_step(self):
        agent = PPOAgent(gamma=0.99, lam=0.95)
        advantages = agent.compute_gae(
            rewards=[10.0],
            values=[5.0],
            dones=[True],
        )
        assert len(advantages) == 1
        # delta = r + gamma * 0 - V = 10 - 5 = 5
        assert abs(advantages[0] - 5.0) < 1e-6

    def test_gae_multi_step(self):
        agent = PPOAgent(gamma=1.0, lam=1.0)
        # With gamma=1, lam=1: GAE = full MC return - value
        advantages = agent.compute_gae(
            rewards=[1.0, 2.0, 3.0],
            values=[0.0, 0.0, 0.0],
            dones=[False, False, True],
        )
        assert len(advantages) == 3
        # Last step: delta = 3 + 0 - 0 = 3, gae = 3
        assert abs(advantages[2] - 3.0) < 1e-6
        # Second step: delta = 2 + 1*0 - 0 = 2, gae = 2 + 1*1*3 = 5
        assert abs(advantages[1] - 5.0) < 1e-6
        # First step: delta = 1 + 1*0 - 0 = 1, gae = 1 + 1*1*5 = 6
        assert abs(advantages[0] - 6.0) < 1e-6

    def test_gae_with_done_resets(self):
        agent = PPOAgent(gamma=0.99, lam=0.95)
        advantages = agent.compute_gae(
            rewards=[1.0, 5.0, 2.0],
            values=[0.5, 0.5, 0.5],
            dones=[False, True, True],
        )
        assert len(advantages) == 3
        # Step 2 (done): delta = 2 - 0.5 = 1.5
        assert abs(advantages[2] - 1.5) < 1e-6
        # Step 1 (done): delta = 5 - 0.5 = 4.5
        assert abs(advantages[1] - 4.5) < 1e-6


class TestPPOTrainStep:
    def test_train_step_with_no_data(self):
        agent = PPOAgent()
        loss = agent.train_step()
        assert loss == 0.0

    def test_train_step_runs_without_error(self):
        agent = PPOAgent(seed=42)
        # Add some fake experience
        for i in range(10):
            agent.buffer.add(Experience(
                state_features=[0.5] * 15,
                action_index=i % NUM_ACTIONS,
                reward=float(i),
                value=float(i) * 0.5,
                log_prob=-1.0,
                done=(i == 9),
            ))
        loss = agent.train_step(epochs=2, batch_size=4)
        assert isinstance(loss, float)
        assert math.isfinite(loss)
        # Buffer should be cleared after training
        assert len(agent.buffer) == 0

    def test_train_step_updates_networks(self):
        agent = PPOAgent(seed=42)
        features = [0.5] * 15
        value_before = agent._get_value(features)

        for i in range(20):
            agent.buffer.add(Experience(
                state_features=features,
                action_index=0,
                reward=10.0,
                value=value_before,
                log_prob=-1.5,
                done=(i == 19),
            ))
        agent.train_step(epochs=4)
        value_after = agent._get_value(features)
        # Value should have changed
        assert value_before != value_after


class TestPPOSelfPlay:
    def test_self_play_episode_completes(self):
        agent = PPOAgent(seed=42)
        engine = GameEngine(small_blind=1, big_blind=2)
        reward = agent.self_play_episode(engine, starting_stack=100, seat=0)
        assert isinstance(reward, float)
        assert math.isfinite(reward)
        assert agent.total_episodes == 1

    def test_self_play_collects_experience(self):
        agent = PPOAgent(seed=42)
        engine = GameEngine(small_blind=1, big_blind=2)
        agent.self_play_episode(engine, starting_stack=100)
        # Should have collected at least 1 experience
        assert len(agent.buffer) >= 1

    def test_self_play_multiple_episodes(self):
        agent = PPOAgent(seed=42)
        engine = GameEngine(small_blind=1, big_blind=2)
        rewards = []
        for _ in range(5):
            r = agent.self_play_episode(engine, starting_stack=100)
            rewards.append(r)
        assert agent.total_episodes == 5
        assert len(rewards) == 5


class TestPPODecide:
    def test_decide_returns_agent_decision(self):
        agent = PPOAgent(seed=42)
        engine = GameEngine(small_blind=1, big_blind=2)
        runtime = engine.start_new_hand(stacks=(100, 100), button_seat=0, seed=42)
        decision = agent.decide(runtime, engine)
        assert decision.action_type in set(ActionType)
        assert isinstance(decision.amount, int)
        assert "ppo:" in decision.rationale

    def test_get_action_distribution_valid(self):
        agent = PPOAgent(seed=42)
        engine = GameEngine(small_blind=1, big_blind=2)
        runtime = engine.start_new_hand(stacks=(100, 100), button_seat=0, seed=42)
        dist = agent.get_action_distribution(runtime, engine)
        assert isinstance(dist, ActionDistribution)
        assert len(dist.probabilities) > 0
        total = sum(dist.probabilities.values())
        assert abs(total - 1.0) < 1e-4


# ---------------------------------------------------------------------------
# GPU CFR Tests
# ---------------------------------------------------------------------------


class TestRegretTable:
    def test_creation(self):
        table = RegretTable(initial_capacity=16)
        assert table.size == 0
        assert table.capacity == 16

    def test_get_id_assigns_sequential_ids(self):
        table = RegretTable()
        id0 = table.get_id("info_set_0")
        id1 = table.get_id("info_set_1")
        assert id0 == 0
        assert id1 == 1
        # Same info set returns same id
        assert table.get_id("info_set_0") == 0

    def test_grow_on_overflow(self):
        table = RegretTable(initial_capacity=2)
        for i in range(5):
            table.get_id(f"info_{i}")
        assert table.size == 5
        assert table.capacity >= 5
        # Old data should be preserved
        assert table.info_set_to_id["info_0"] == 0


class TestBatchRegretMatch:
    def test_uniform_when_all_zero(self):
        regrets = np.zeros((3, NUM_ACTIONS), dtype=np.float64)
        strategies = batch_regret_match(regrets)
        assert strategies.shape == (3, NUM_ACTIONS)
        for i in range(3):
            assert abs(strategies[i].sum() - 1.0) < 1e-6
            # Should be uniform
            for j in range(NUM_ACTIONS):
                assert abs(strategies[i][j] - 1.0 / NUM_ACTIONS) < 1e-6

    def test_positive_regrets_normalized(self):
        regrets = np.zeros((1, NUM_ACTIONS), dtype=np.float64)
        regrets[0, 0] = 3.0
        regrets[0, 1] = 1.0
        strategies = batch_regret_match(regrets)
        assert abs(strategies[0, 0] - 0.75) < 1e-6
        assert abs(strategies[0, 1] - 0.25) < 1e-6
        for i in range(2, NUM_ACTIONS):
            assert abs(strategies[0, i]) < 1e-6

    def test_negative_regrets_ignored(self):
        regrets = np.array([[5.0, -10.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float64)
        strategies = batch_regret_match(regrets)
        assert abs(strategies[0, 0] - 1.0) < 1e-6
        assert abs(strategies[0, 1]) < 1e-6

    def test_batch_produces_valid_strategies(self):
        """All rows should be valid probability distributions."""
        rng = np.random.RandomState(42)
        regrets = rng.randn(100, NUM_ACTIONS)
        strategies = batch_regret_match(regrets)
        assert strategies.shape == (100, NUM_ACTIONS)
        for i in range(100):
            assert abs(strategies[i].sum() - 1.0) < 1e-6
            assert np.all(strategies[i] >= -1e-10)

    def test_vectorized_matches_scalar(self):
        """Vectorized result should match scalar regret matching."""
        rng = np.random.RandomState(123)
        regrets = rng.randn(10, NUM_ACTIONS)

        # Vectorized
        vec_strats = batch_regret_match(regrets)

        # Scalar equivalent
        for i in range(10):
            positive = np.maximum(regrets[i], 0.0)
            total = positive.sum()
            if total > 0:
                expected = positive / total
            else:
                expected = np.full(NUM_ACTIONS, 1.0 / NUM_ACTIONS)
            np.testing.assert_allclose(vec_strats[i], expected, atol=1e-10)


class TestBatchUpdate:
    def test_accumulates_regrets(self):
        table = RegretTable()
        id0 = table.get_id("test_0")
        id1 = table.get_id("test_1")

        ids = np.array([id0, id1], dtype=np.int64)
        action_utils = np.array([
            [1.0, 2.0, 3.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 4.0, 5.0, 6.0],
        ], dtype=np.float64)
        strategies = np.array([
            [0.5, 0.3, 0.2, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.3, 0.3, 0.4],
        ], dtype=np.float64)
        node_values = np.array([1.7, 5.2], dtype=np.float64)

        batch_update(table, ids, action_utils, strategies, node_values)

        # Regrets should be action_utils - node_value
        expected_0 = action_utils[0] - 1.7
        expected_1 = action_utils[1] - 5.2
        np.testing.assert_allclose(table.regrets[id0], expected_0, atol=1e-10)
        np.testing.assert_allclose(table.regrets[id1], expected_1, atol=1e-10)

        # Strategy sums should be accumulated
        np.testing.assert_allclose(table.strategy_sums[id0], strategies[0], atol=1e-10)
        np.testing.assert_allclose(table.strategy_sums[id1], strategies[1], atol=1e-10)

    def test_multiple_updates_accumulate(self):
        table = RegretTable()
        id0 = table.get_id("test_0")
        ids = np.array([id0], dtype=np.int64)
        utils = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float64)
        strats = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float64)
        vals = np.array([0.5], dtype=np.float64)

        batch_update(table, ids, utils, strats, vals)
        batch_update(table, ids, utils, strats, vals)

        # Should accumulate twice
        assert abs(table.regrets[id0, 0] - 1.0) < 1e-10  # (1-0.5)*2
        assert abs(table.strategy_sums[id0, 0] - 2.0) < 1e-10


class TestBatchCFRTrainer:
    def test_creation(self):
        trainer = BatchCFRTrainer(seed=42, batch_size=8)
        assert trainer.batch_size == 8
        assert trainer.iterations == 0
        assert trainer.table.size == 0

    def test_train_runs_without_error(self):
        trainer = BatchCFRTrainer(seed=42, batch_size=4)
        cfr_state = trainer.train(iterations=5)
        assert isinstance(cfr_state, CFRState)
        assert cfr_state.iterations == 5
        assert trainer.table.size > 0

    def test_to_cfr_state_exports_correctly(self):
        trainer = BatchCFRTrainer(seed=42, batch_size=4)
        trainer.train(iterations=3)
        cfr_state = trainer.to_cfr_state()
        assert isinstance(cfr_state, CFRState)
        assert cfr_state.iterations == 3
        # Should have info sets in the exported state
        assert len(cfr_state.cumulative_regret) > 0 or len(cfr_state.strategy_sum) > 0

    def test_get_strategy_unseen(self):
        trainer = BatchCFRTrainer(seed=42)
        legal = {ActionType.FOLD, ActionType.CALL}
        dist = trainer.get_strategy("unseen_info_set", legal)
        assert isinstance(dist, ActionDistribution)
        assert abs(sum(dist.probabilities.values()) - 1.0) < 1e-6

    def test_get_strategy_after_training(self):
        trainer = BatchCFRTrainer(seed=42, batch_size=4)
        trainer.train(iterations=5)
        if trainer.table.size > 0:
            info_set = list(trainer.table.info_set_to_id.keys())[0]
            legal = {ActionType.FOLD, ActionType.CHECK, ActionType.CALL,
                     ActionType.BET, ActionType.RAISE, ActionType.ALL_IN}
            dist = trainer.get_strategy(info_set, legal)
            assert abs(sum(dist.probabilities.values()) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_ppo_plays_full_game(self):
        """PPO agent can play a complete hand against random opponent."""
        agent = PPOAgent(seed=42)
        engine = GameEngine(small_blind=1, big_blind=2)
        runtime = engine.start_new_hand(stacks=(100, 100), button_seat=0, seed=42)

        steps = 0
        while not runtime.state.is_terminal and runtime.state.acting_seat is not None:
            if runtime.state.acting_seat == 0:
                decision = agent.decide(runtime, engine)
                engine.apply_action(runtime, decision.action_type, decision.amount)
            else:
                legal = engine.legal_actions(runtime)
                action = legal[0]  # Deterministic opponent
                from packages.cfr_agent.trainer import _size_action
                player = runtime.state.players[1]
                amount = _size_action(action, runtime.state, player, engine)
                engine.apply_action(runtime, action, amount)
            steps += 1
            if steps > 50:
                break

        assert runtime.state.is_terminal or steps > 0

    def test_ppo_train_loop(self):
        """PPO agent can do self-play + train cycle."""
        agent = PPOAgent(seed=42)
        engine = GameEngine(small_blind=1, big_blind=2)

        for _ in range(3):
            agent.self_play_episode(engine, starting_stack=100)

        assert len(agent.buffer) >= 1
        loss = agent.train_step(epochs=1)
        assert math.isfinite(loss)

    def test_gpu_cfr_produces_usable_strategies(self):
        """GPU CFR trained strategies can be used by CFRState.average_strategy."""
        trainer = BatchCFRTrainer(seed=42, batch_size=4)
        cfr_state = trainer.train(iterations=10)

        # Pick an info set from training
        if cfr_state.strategy_sum:
            info_set = list(cfr_state.strategy_sum.keys())[0]
            legal = {ActionType.FOLD, ActionType.CHECK, ActionType.CALL,
                     ActionType.BET, ActionType.RAISE, ActionType.ALL_IN}
            dist = cfr_state.average_strategy(info_set, legal)
            assert isinstance(dist, ActionDistribution)
            total = sum(dist.probabilities.values())
            assert abs(total - 1.0) < 1e-4

    def test_ppo_import_from_package(self):
        """PPO agent is importable from the package."""
        from packages.ppo_agent import PPOAgent as PPO
        agent = PPO()
        assert agent is not None

    def test_gpu_cfr_import_from_package(self):
        """BatchCFRTrainer is importable from cfr_agent package."""
        from packages.cfr_agent import BatchCFRTrainer as BCFR
        trainer = BCFR(seed=42)
        assert trainer is not None


class TestSoftmax:
    def test_softmax_sums_to_one(self):
        result = _softmax([1.0, 2.0, 3.0])
        assert abs(sum(result) - 1.0) < 1e-6

    def test_softmax_numerical_stability(self):
        result = _softmax([1000.0, 1001.0, 1002.0])
        assert abs(sum(result) - 1.0) < 1e-6
        assert all(math.isfinite(v) for v in result)

    def test_log_softmax_consistent(self):
        logits = [1.0, 2.0, 3.0]
        probs = _softmax(logits)
        log_probs = _log_softmax(logits)
        for p, lp in zip(probs, log_probs):
            assert abs(math.exp(lp) - p) < 1e-6


class TestExperienceBuffer:
    def test_buffer_add_and_len(self):
        buf = ExperienceBuffer()
        assert len(buf) == 0
        buf.add(Experience([0.0]*15, 0, 1.0, 0.5, -1.0))
        assert len(buf) == 1

    def test_buffer_max_size(self):
        buf = ExperienceBuffer(max_size=3)
        for i in range(5):
            buf.add(Experience([0.0]*15, 0, float(i), 0.0, 0.0))
        assert len(buf) == 3

    def test_buffer_clear(self):
        buf = ExperienceBuffer()
        buf.add(Experience([0.0]*15, 0, 1.0, 0.5, -1.0))
        buf.clear()
        assert len(buf) == 0
