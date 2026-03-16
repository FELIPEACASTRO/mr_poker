"""Tests for Batch 7: Embedding CFR, HDCFR, QRE.

~30 tests covering:
- EmbeddingCFR: creation, encoder, clustering, strategy, update
- HDCFR: creation, skill selection, executor, combined policy, training
- QRE: creation, strategy computation, temperature effects, annealing, export
- Integration: all three produce compatible ActionDistribution
"""

from __future__ import annotations

import math

import pytest

from packages.common.types import ActionType
from packages.strategy.mixed import ActionDistribution
from packages.cfr_agent.deep_cfr import FEATURE_DIM, NUM_ACTIONS, ACTION_INDEX
from packages.cfr_agent.trainer import CFRState

from packages.cfr_agent.embedding_cfr import (
    EmbeddingCFRTrainer,
    EmbeddingRegretTable,
    ClusterEntry,
    _euclidean_distance,
)
from packages.cfr_agent.hdcfr import (
    HDCFRTrainer,
    HierarchicalPolicy,
    Skill,
    NUM_SKILLS,
    SkillMemory,
    _softmax,
    _one_hot,
)
from packages.cfr_agent.qre import (
    QRESolver,
    QREState,
)


LEGAL_ACTIONS = {ActionType.FOLD, ActionType.CALL, ActionType.RAISE}
SAMPLE_FEATURES = [0.5] * FEATURE_DIM


# ─── EmbeddingCFR Tests ────────────────────────────────────────────────

class TestEmbeddingCFRCreation:
    def test_creation_defaults(self):
        trainer = EmbeddingCFRTrainer()
        assert trainer.embedding_dim == 16
        assert trainer.iterations == 0
        assert trainer.regret_table.num_clusters == 0

    def test_creation_custom_params(self):
        trainer = EmbeddingCFRTrainer(embedding_dim=8, threshold=1.0, hidden_dim=16)
        assert trainer.embedding_dim == 8
        assert trainer.regret_table.threshold == 1.0


class TestEmbeddingCFREncoder:
    def test_encoder_produces_embedding(self):
        trainer = EmbeddingCFRTrainer(embedding_dim=16)
        embedding = trainer.encode(SAMPLE_FEATURES)
        assert len(embedding) == 16
        assert all(isinstance(v, float) for v in embedding)

    def test_encoder_different_inputs_different_outputs(self):
        trainer = EmbeddingCFRTrainer(embedding_dim=16, seed=42)
        emb1 = trainer.encode([1.0] * FEATURE_DIM)
        emb2 = trainer.encode([0.0] * FEATURE_DIM)
        # Different inputs should produce different embeddings
        assert emb1 != emb2


class TestEmbeddingCFRClustering:
    def test_similar_features_cluster_together(self):
        trainer = EmbeddingCFRTrainer(threshold=5.0, seed=42)
        features1 = [0.5] * FEATURE_DIM
        features2 = [0.51] * FEATURE_DIM  # very similar
        cid1 = trainer._cluster_info_set(features1)
        cid2 = trainer._cluster_info_set(features2)
        assert cid1 == cid2, "Similar features should map to the same cluster"

    def test_different_features_separate_clusters(self):
        trainer = EmbeddingCFRTrainer(threshold=0.01, seed=42)
        features1 = [0.0] * FEATURE_DIM
        features2 = [1.0] * FEATURE_DIM
        cid1 = trainer._cluster_info_set(features1)
        cid2 = trainer._cluster_info_set(features2)
        assert cid1 != cid2, "Very different features should get different clusters"

    def test_cluster_count_grows(self):
        trainer = EmbeddingCFRTrainer(threshold=0.01, seed=42)
        for i in range(5):
            features = [float(i)] * FEATURE_DIM
            trainer._cluster_info_set(features)
        assert trainer.regret_table.num_clusters >= 2


class TestEmbeddingCFRStrategy:
    def test_current_strategy_valid_distribution(self):
        trainer = EmbeddingCFRTrainer()
        strategy = trainer.current_strategy(SAMPLE_FEATURES, LEGAL_ACTIONS)
        assert isinstance(strategy, ActionDistribution)
        total = sum(strategy.probabilities.values())
        assert abs(total - 1.0) < 1e-6
        for action in LEGAL_ACTIONS:
            assert action in strategy.probabilities

    def test_current_strategy_uniform_initially(self):
        trainer = EmbeddingCFRTrainer()
        strategy = trainer.current_strategy(SAMPLE_FEATURES, LEGAL_ACTIONS)
        for prob in strategy.probabilities.values():
            assert abs(prob - 1.0 / len(LEGAL_ACTIONS)) < 1e-6


class TestEmbeddingCFRUpdate:
    def test_update_modifies_state(self):
        trainer = EmbeddingCFRTrainer()
        strategy = trainer.current_strategy(SAMPLE_FEATURES, LEGAL_ACTIONS)
        utilities = {ActionType.FOLD: -1.0, ActionType.CALL: 0.5, ActionType.RAISE: 1.5}
        ev = sum(strategy.probabilities.get(a, 0.0) * u for a, u in utilities.items())
        trainer.update(SAMPLE_FEATURES, strategy, utilities, ev)
        cid = trainer._cluster_info_set(SAMPLE_FEATURES)
        cluster = trainer.regret_table.clusters[cid]
        assert cluster.count >= 1

    def test_train_returns_strategies(self):
        trainer = EmbeddingCFRTrainer(seed=42)
        strategies = trainer.train(iterations=50)
        assert len(strategies) > 0
        for cid, strat in strategies.items():
            assert isinstance(strat, ActionDistribution)

    def test_to_cfr_state_export(self):
        trainer = EmbeddingCFRTrainer(seed=42)
        trainer.train(iterations=20)
        state = trainer.to_cfr_state()
        assert isinstance(state, CFRState)
        assert state.iterations == trainer.iterations
        assert len(state.cumulative_regret) > 0


# ─── HDCFR Tests ───────────────────────────────────────────────────────

class TestHDCFRCreation:
    def test_creation_defaults(self):
        trainer = HDCFRTrainer()
        assert trainer.num_skills == NUM_SKILLS
        assert trainer.iterations == 0

    def test_policy_has_correct_structure(self):
        trainer = HDCFRTrainer()
        assert len(trainer.policy.executors) == NUM_SKILLS


class TestHDCFRSkillSelection:
    def test_skill_selection_returns_valid_probabilities(self):
        policy = HierarchicalPolicy()
        skill_id, probs = policy.select_skill(SAMPLE_FEATURES)
        assert 0 <= skill_id < NUM_SKILLS
        assert len(probs) == NUM_SKILLS
        assert abs(sum(probs) - 1.0) < 1e-6
        assert all(p >= 0 for p in probs)

    def test_skill_enum_values(self):
        assert Skill.AGGRESSIVE == 0
        assert Skill.PASSIVE == 1
        assert Skill.TRAPPING == 2
        assert Skill.BALANCED == 3


class TestHDCFRExecutor:
    def test_executor_produces_valid_action_probs(self):
        policy = HierarchicalPolicy()
        for skill_id in range(NUM_SKILLS):
            probs = policy.execute_skill(SAMPLE_FEATURES, skill_id)
            assert len(probs) == NUM_ACTIONS
            assert abs(sum(probs) - 1.0) < 1e-6
            assert all(p >= 0 for p in probs)


class TestHDCFRCombinedPolicy:
    def test_combined_policy_sums_to_one(self):
        policy = HierarchicalPolicy()
        action_probs = policy.get_action_probs(SAMPLE_FEATURES)
        assert len(action_probs) == NUM_ACTIONS
        assert abs(sum(action_probs) - 1.0) < 1e-6

    def test_get_strategy_valid_distribution(self):
        trainer = HDCFRTrainer()
        strategy = trainer.get_strategy(SAMPLE_FEATURES, LEGAL_ACTIONS)
        assert isinstance(strategy, ActionDistribution)
        total = sum(strategy.probabilities.values())
        assert abs(total - 1.0) < 1e-6


class TestHDCFRTraining:
    def test_train_skills_runs(self):
        trainer = HDCFRTrainer(seed=42)
        trainer.train_skills(iterations=30)
        assert trainer.iterations == 30

    def test_skill_memory_stores_entries(self):
        memory = SkillMemory()
        memory.add([0.5] * FEATURE_DIM, 0, 1.0)
        memory.add([0.5] * FEATURE_DIM, 1, -0.5)
        assert memory.size(0) == 1
        assert memory.size(1) == 1

    def test_to_cfr_state_export(self):
        trainer = HDCFRTrainer(seed=42)
        trainer.train_skills(iterations=10)
        state = trainer.to_cfr_state()
        assert isinstance(state, CFRState)
        assert len(state.strategy_sum) == NUM_SKILLS

    def test_hierarchical_decomposition(self):
        """Different skills should produce different action distributions."""
        policy = HierarchicalPolicy(seed=42)
        probs_by_skill = []
        for skill_id in range(NUM_SKILLS):
            probs = policy.execute_skill(SAMPLE_FEATURES, skill_id)
            probs_by_skill.append(probs)
        # At least some skills should differ (different random init)
        all_same = all(probs_by_skill[i] == probs_by_skill[0] for i in range(1, NUM_SKILLS))
        assert not all_same, "Skills should produce different action distributions"


# ─── QRE Tests ─────────────────────────────────────────────────────────

class TestQRECreation:
    def test_creation_defaults(self):
        solver = QRESolver()
        assert solver.lambda_ == 0.1
        assert solver.iterations == 0

    def test_creation_custom_lambda(self):
        solver = QRESolver(lambda_=0.5)
        assert solver.lambda_ == 0.5


class TestQREStrategy:
    def test_compute_qre_strategy_valid_distribution(self):
        solver = QRESolver()
        utilities = {ActionType.FOLD: -1.0, ActionType.CALL: 0.5, ActionType.RAISE: 1.5}
        strategy = solver.compute_qre_strategy("test", utilities, LEGAL_ACTIONS)
        assert isinstance(strategy, ActionDistribution)
        total = sum(strategy.probabilities.values())
        assert abs(total - 1.0) < 1e-6

    def test_high_temperature_approaches_uniform(self):
        solver = QRESolver(lambda_=100.0)
        utilities = {ActionType.FOLD: -10.0, ActionType.CALL: 0.0, ActionType.RAISE: 10.0}
        strategy = solver.compute_qre_strategy("test", utilities, LEGAL_ACTIONS)
        # With very high temperature, all probs should be nearly equal
        expected = 1.0 / len(LEGAL_ACTIONS)
        for prob in strategy.probabilities.values():
            assert abs(prob - expected) < 0.05

    def test_low_temperature_approaches_greedy(self):
        solver = QRESolver(lambda_=0.001)
        utilities = {ActionType.FOLD: -10.0, ActionType.CALL: 0.0, ActionType.RAISE: 10.0}
        strategy = solver.compute_qre_strategy("test", utilities, LEGAL_ACTIONS)
        # With very low temperature, RAISE should dominate
        assert strategy.probabilities[ActionType.RAISE] > 0.99

    def test_zero_lambda_is_greedy(self):
        solver = QRESolver(lambda_=0.0)
        utilities = {ActionType.FOLD: -1.0, ActionType.CALL: 0.5, ActionType.RAISE: 1.5}
        strategy = solver.compute_qre_strategy("test", utilities, LEGAL_ACTIONS)
        assert strategy.probabilities[ActionType.RAISE] == 1.0


class TestQREUtilities:
    def test_update_utilities(self):
        solver = QRESolver()
        solver.update_utilities("info1", {ActionType.FOLD: -1.0, ActionType.CALL: 0.5})
        assert "info1" in solver.state.utilities
        assert solver.state.utilities["info1"]["fold"] == -1.0
        assert solver.state.utilities["info1"]["call"] == 0.5

    def test_update_utilities_running_average(self):
        solver = QRESolver()
        solver.update_utilities("info1", {ActionType.CALL: 2.0})
        solver.update_utilities("info1", {ActionType.CALL: 4.0})
        # Running average of 2.0 and 4.0 = 3.0
        assert abs(solver.state.utilities["info1"]["call"] - 3.0) < 1e-6


class TestQREAnnealing:
    def test_lambda_annealing(self):
        solver = QRESolver(lambda_=0.1, seed=42)
        strategies = solver.iterative_solve(
            num_iterations=50, initial_lambda=0.5, target_lambda=0.1
        )
        assert solver.iterations == 50
        assert len(strategies) > 0

    def test_iterative_solve_produces_strategies(self):
        solver = QRESolver(seed=42)
        strategies = solver.iterative_solve(num_iterations=30)
        for info_set, strat in strategies.items():
            assert isinstance(strat, ActionDistribution)
            total = sum(strat.probabilities.values())
            assert abs(total - 1.0) < 1e-6


class TestQREExport:
    def test_to_cfr_state_export(self):
        solver = QRESolver(seed=42)
        solver.iterative_solve(num_iterations=20)
        state = solver.to_cfr_state()
        assert isinstance(state, CFRState)
        assert state.iterations == solver.iterations
        assert len(state.strategy_sum) > 0

    def test_to_cfr_state_valid_strategies(self):
        solver = QRESolver(seed=42)
        solver.update_utilities("test_is", {
            ActionType.FOLD: -1.0,
            ActionType.CALL: 0.5,
            ActionType.RAISE: 1.5,
        })
        solver.iterations = 10
        state = solver.to_cfr_state()
        assert "test_is" in state.strategy_sum
        # Strategy sum values should be non-negative
        for val in state.strategy_sum["test_is"].values():
            assert val >= 0


# ─── Integration Tests ─────────────────────────────────────────────────

class TestIntegration:
    def test_all_produce_action_distribution(self):
        """All three methods should produce compatible ActionDistribution."""
        emb = EmbeddingCFRTrainer(seed=42)
        hdcfr = HDCFRTrainer(seed=42)
        qre = QRESolver(seed=42)

        # Set up QRE with some utilities
        qre.update_utilities("test", {a: 0.0 for a in LEGAL_ACTIONS})

        strats = [
            emb.current_strategy(SAMPLE_FEATURES, LEGAL_ACTIONS),
            hdcfr.get_strategy(SAMPLE_FEATURES, LEGAL_ACTIONS),
            qre.get_strategy("test", LEGAL_ACTIONS),
        ]

        for strat in strats:
            assert isinstance(strat, ActionDistribution)
            total = sum(strat.probabilities.values())
            assert abs(total - 1.0) < 1e-6
            assert all(p >= 0 for p in strat.probabilities.values())

    def test_all_export_to_cfr_state(self):
        """All three should export to CFRState."""
        emb = EmbeddingCFRTrainer(seed=42)
        emb.train(iterations=10)

        hdcfr = HDCFRTrainer(seed=42)
        hdcfr.train_skills(iterations=10)

        qre = QRESolver(seed=42)
        qre.iterative_solve(num_iterations=10)

        for solver in [emb, hdcfr, qre]:
            state = solver.to_cfr_state()
            assert isinstance(state, CFRState)
            assert state.iterations > 0

    def test_softmax_helper(self):
        """Verify _softmax used in HDCFR."""
        probs = _softmax([1.0, 2.0, 3.0])
        assert abs(sum(probs) - 1.0) < 1e-6
        assert probs[2] > probs[1] > probs[0]

    def test_one_hot_helper(self):
        vec = _one_hot(2, 4)
        assert vec == [0.0, 0.0, 1.0, 0.0]

    def test_euclidean_distance(self):
        assert _euclidean_distance([0, 0], [3, 4]) == 5.0
        assert _euclidean_distance([1, 1], [1, 1]) == 0.0
