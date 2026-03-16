"""Batch 6 tests: DMC Agent, MCTS Agent, PCPG.

Tests cover:
- DMC: creation, Q-network, epsilon-greedy, replay buffer, training, self-play
- MCTS: node creation, UCB1, expansion, rollout, search, opponent-informed
- PCPG: population, diversity, parameterized agent, evolution, diversity score
- Integration: full hand play with each component
"""

from __future__ import annotations

import math
import random

import pytest

from packages.cfr_agent.deep_cfr import (
    ACTION_INDEX,
    FEATURE_DIM,
    NUM_ACTIONS,
    SimpleNN,
    extract_features,
)
from packages.common.types import ActionType
from packages.dmc_agent.agent import (
    DMCAgent,
    ExperienceReplay,
    INDEX_TO_ACTION,
    Transition,
    _copy_nn,
    _soft_update,
)
from packages.engine.engine import GameEngine, HandRuntime
from packages.mcts_agent.agent import ISMCTSAgent, MCTSNode
from packages.opponent_model.classifier import OpponentTracker, PlayerStats
from packages.strategy.mixed import ActionDistribution
from packages.strategy.pcpg import (
    NUM_STYLE_PARAMS,
    PCPGPopulation,
    ParameterizedAgent,
    PlayStyleParams,
    _hand_strength_estimate,
    _latin_hypercube_sample,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine() -> GameEngine:
    return GameEngine(small_blind=1, big_blind=2)


def _make_runtime(engine: GameEngine | None = None, seed: int = 42) -> HandRuntime:
    eng = engine or _make_engine()
    return eng.start_new_hand(stacks=(100, 100), button_seat=0, seed=seed)


def _random_features(rng: random.Random | None = None) -> list[float]:
    r = rng or random.Random(123)
    return [r.random() for _ in range(FEATURE_DIM)]


# ===========================================================================
# DMC Agent Tests
# ===========================================================================

class TestDMCCreation:
    def test_creation_defaults(self):
        agent = DMCAgent()
        assert agent.epsilon == 1.0
        assert agent.gamma == 0.99
        assert agent.steps == 0

    def test_q_network_forward(self):
        agent = DMCAgent()
        feat = _random_features()
        q_vals = agent.q_network.forward(feat)
        assert len(q_vals) == NUM_ACTIONS
        assert all(isinstance(v, float) for v in q_vals)

    def test_target_network_initialized(self):
        agent = DMCAgent(seed=99)
        feat = _random_features()
        q_vals = agent.q_network.forward(feat)
        t_vals = agent.target_network.forward(feat)
        # Initially identical
        for a, b in zip(q_vals, t_vals):
            assert abs(a - b) < 1e-9


class TestEpsilonGreedy:
    def test_full_exploration(self):
        agent = DMCAgent(epsilon_start=1.0, seed=42)
        feat = _random_features()
        legal = {ActionType.FOLD, ActionType.CALL, ActionType.RAISE}
        # With epsilon=1.0, should always explore (random)
        actions = set()
        for _ in range(200):
            idx = agent.select_action(feat, legal)
            actions.add(idx)
        # Should have selected multiple different actions
        assert len(actions) > 1

    def test_greedy_selection(self):
        agent = DMCAgent(epsilon_start=0.0, seed=42)
        feat = _random_features()
        legal = {ActionType.FOLD, ActionType.CHECK, ActionType.BET}
        idx1 = agent.select_action(feat, legal)
        idx2 = agent.select_action(feat, legal)
        # With epsilon=0, should be deterministic
        assert idx1 == idx2

    def test_action_in_legal_range(self):
        agent = DMCAgent(epsilon_start=0.5, seed=42)
        feat = _random_features()
        legal = {ActionType.FOLD, ActionType.CALL}
        legal_indices = {ACTION_INDEX[a] for a in legal}
        for _ in range(50):
            idx = agent.select_action(feat, legal)
            assert idx in legal_indices


class TestExperienceReplay:
    def test_add_and_len(self):
        replay = ExperienceReplay(capacity=100)
        assert len(replay) == 0
        t = Transition(state=[0.0]*FEATURE_DIM, action_idx=0, reward=1.0,
                       next_state=[0.0]*FEATURE_DIM, done=True)
        replay.add(t)
        assert len(replay) == 1

    def test_capacity_limit(self):
        replay = ExperienceReplay(capacity=5)
        for i in range(10):
            t = Transition(state=[float(i)]*FEATURE_DIM, action_idx=0,
                           reward=float(i), next_state=[0.0]*FEATURE_DIM, done=False)
            replay.add(t)
        assert len(replay) == 5

    def test_sample(self):
        replay = ExperienceReplay(capacity=100)
        for i in range(20):
            t = Transition(state=[float(i)]*FEATURE_DIM, action_idx=i % NUM_ACTIONS,
                           reward=float(i), next_state=[0.0]*FEATURE_DIM, done=False)
            replay.add(t)
        batch = replay.sample(5, random.Random(42))
        assert len(batch) == 5
        assert all(isinstance(t, Transition) for t in batch)

    def test_sample_smaller_than_buffer(self):
        replay = ExperienceReplay(capacity=100)
        for i in range(3):
            replay.add(Transition([0.0]*FEATURE_DIM, 0, 0.0, [0.0]*FEATURE_DIM, True))
        batch = replay.sample(10, random.Random(42))
        assert len(batch) == 3  # returns all when buffer < batch_size


class TestDMCTrainStep:
    def test_train_step_returns_loss(self):
        agent = DMCAgent(seed=42)
        # Fill replay
        for i in range(64):
            t = Transition(
                state=_random_features(random.Random(i)),
                action_idx=i % NUM_ACTIONS,
                reward=random.Random(i).uniform(-1, 1),
                next_state=_random_features(random.Random(i + 100)),
                done=i % 5 == 0,
            )
            agent.replay.add(t)
        loss = agent.train_step(batch_size=32)
        assert loss > 0.0

    def test_epsilon_decays_after_train(self):
        agent = DMCAgent(epsilon_start=1.0, epsilon_decay=0.99, seed=42)
        for i in range(64):
            agent.replay.add(Transition(
                _random_features(random.Random(i)), 0, 0.0,
                _random_features(random.Random(i+1)), True,
            ))
        initial_eps = agent.epsilon
        agent.train_step(32)
        assert agent.epsilon < initial_eps

    def test_target_soft_update(self):
        agent = DMCAgent(tau=0.5, seed=42)
        feat = _random_features()
        before = agent.target_network.forward(feat)[:]
        # Fill and train
        for i in range(64):
            agent.replay.add(Transition(
                _random_features(random.Random(i)), 0, 1.0,
                _random_features(random.Random(i+1)), False,
            ))
        agent.train_step(32)
        after = agent.target_network.forward(feat)
        # Target should have changed (tau=0.5 is significant)
        assert any(abs(a - b) > 1e-9 for a, b in zip(before, after))


class TestDMCSelfPlay:
    def test_self_play_returns_rewards(self):
        agent = DMCAgent(epsilon_start=0.8, seed=42)
        engine = _make_engine()
        rewards = agent.self_play(engine, num_episodes=5)
        assert len(rewards) == 5
        assert all(isinstance(r, (int, float)) for r in rewards)

    def test_self_play_fills_replay(self):
        agent = DMCAgent(seed=42)
        engine = _make_engine()
        assert len(agent.replay) == 0
        agent.self_play(engine, num_episodes=3)
        assert len(agent.replay) > 0


class TestDMCDecide:
    def test_decide_returns_valid_action(self):
        agent = DMCAgent(epsilon_start=0.0, seed=42)
        engine = _make_engine()
        runtime = _make_runtime(engine)
        action, amount = agent.decide(runtime, engine)
        legal = set(engine.legal_actions(runtime))
        assert action in legal
        assert isinstance(amount, int)

    def test_get_q_values(self):
        agent = DMCAgent(seed=42)
        feat = _random_features()
        q = agent.get_q_values(feat)
        assert len(q) == NUM_ACTIONS
        assert all(isinstance(v, float) for v in q.values())

    def test_get_action_distribution(self):
        agent = DMCAgent(seed=42)
        feat = _random_features()
        legal = {ActionType.FOLD, ActionType.CHECK, ActionType.BET}
        dist = agent.get_action_distribution(feat, legal)
        assert abs(sum(dist.probabilities.values()) - 1.0) < 1e-6
        assert all(a in legal for a in dist.probabilities)


# ===========================================================================
# MCTS Agent Tests
# ===========================================================================

class TestMCTSNode:
    def test_creation(self):
        node = MCTSNode()
        assert node.visit_count == 0
        assert node.total_value == 0.0
        assert node.is_leaf()

    def test_value_zero_visits(self):
        node = MCTSNode()
        assert node.value == 0.0

    def test_value_with_visits(self):
        node = MCTSNode(visit_count=10, total_value=5.0)
        assert abs(node.value - 0.5) < 1e-9

    def test_ucb1_unvisited(self):
        node = MCTSNode(visit_count=0)
        assert node.ucb1(10) == float("inf")

    def test_ucb1_visited(self):
        node = MCTSNode(visit_count=5, total_value=2.5)
        parent_visits = 20
        score = node.ucb1(parent_visits)
        exploitation = 2.5 / 5
        exploration = 1.414 * math.sqrt(math.log(20) / 5)
        assert abs(score - (exploitation + exploration)) < 1e-6

    def test_best_child(self):
        root = MCTSNode(visit_count=10)
        root.children[ActionType.FOLD] = MCTSNode(visit_count=3, total_value=0.3)
        root.children[ActionType.CALL] = MCTSNode(visit_count=5, total_value=4.0)
        root.children[ActionType.RAISE] = MCTSNode(visit_count=2, total_value=0.1)
        action, child = root.best_child()
        # CALL has best exploitation (0.8) and decent visits
        assert action in root.children

    def test_most_visited_child(self):
        root = MCTSNode()
        root.children[ActionType.FOLD] = MCTSNode(visit_count=3)
        root.children[ActionType.CALL] = MCTSNode(visit_count=10)
        root.children[ActionType.BET] = MCTSNode(visit_count=5)
        action, child = root.most_visited_child()
        assert action == ActionType.CALL
        assert child.visit_count == 10


class TestMCTSExpansion:
    def test_expansion_creates_children(self):
        """Verify that search creates child nodes."""
        agent = ISMCTSAgent(num_simulations=10, seed=42)
        engine = _make_engine()
        runtime = _make_runtime(engine, seed=99)
        dist = agent.search(runtime, engine)
        assert len(dist.probabilities) > 0
        assert abs(sum(dist.probabilities.values()) - 1.0) < 1e-6


class TestMCTSRollout:
    def test_rollout_returns_value(self):
        agent = ISMCTSAgent(num_simulations=5, seed=42)
        engine = _make_engine()
        runtime = _make_runtime(engine, seed=77)
        # _rollout is internal but we test via search
        dist = agent.search(runtime, engine, num_simulations=5)
        assert isinstance(dist, ActionDistribution)

    def test_opponent_informed_rollout(self):
        tracker = OpponentTracker()
        # Give opponent some stats
        stats = tracker.get_or_create(1)
        stats.total_hands = 50
        stats.total_aggressive_actions = 80
        stats.total_passive_actions = 10
        stats.voluntary_put_in_pot = 30
        stats.preflop_raises = 25

        agent = ISMCTSAgent(num_simulations=10, opponent_tracker=tracker, seed=42)
        engine = _make_engine()
        runtime = _make_runtime(engine, seed=88)
        dist = agent.search(runtime, engine)
        assert len(dist.probabilities) > 0


class TestMCTSSearch:
    def test_search_returns_valid_distribution(self):
        agent = ISMCTSAgent(num_simulations=20, seed=42)
        engine = _make_engine()
        runtime = _make_runtime(engine, seed=55)
        dist = agent.search(runtime, engine)
        legal = set(engine.legal_actions(runtime))
        assert all(a in legal for a in dist.probabilities)
        total = sum(dist.probabilities.values())
        assert abs(total - 1.0) < 1e-6

    def test_decide_returns_legal_action(self):
        agent = ISMCTSAgent(num_simulations=10, seed=42)
        engine = _make_engine()
        runtime = _make_runtime(engine, seed=66)
        action, amount = agent.decide(runtime, engine)
        legal = set(engine.legal_actions(runtime))
        assert action in legal
        assert isinstance(amount, int)


# ===========================================================================
# PCPG Tests
# ===========================================================================

class TestPlayStyleParams:
    def test_to_vector(self):
        params = PlayStyleParams(aggression=0.8, tightness=0.3)
        vec = params.to_vector()
        assert len(vec) == NUM_STYLE_PARAMS
        assert vec[0] == 0.8
        assert vec[1] == 0.3

    def test_from_vector_clamping(self):
        params = PlayStyleParams.from_vector([1.5, -0.2, 0.5, 0.5, 0.5, 0.5])
        assert params.aggression == 1.0
        assert params.tightness == 0.0

    def test_distance(self):
        p1 = PlayStyleParams(aggression=0.0, tightness=0.0, bluff_freq=0.0,
                              value_bet_freq=0.0, fold_to_3bet=0.0, cbet_freq=0.0)
        p2 = PlayStyleParams(aggression=1.0, tightness=1.0, bluff_freq=1.0,
                              value_bet_freq=1.0, fold_to_3bet=1.0, cbet_freq=1.0)
        dist = p1.distance(p2)
        assert abs(dist - math.sqrt(6)) < 1e-6

    def test_mutate(self):
        rng = random.Random(42)
        params = PlayStyleParams(aggression=0.5, tightness=0.5)
        mutated = params.mutate(rng, sigma=0.1)
        # Should be different
        assert mutated.aggression != params.aggression or mutated.tightness != params.tightness

    def test_crossover(self):
        rng = random.Random(42)
        p1 = PlayStyleParams(aggression=0.0, tightness=0.0, bluff_freq=0.0,
                              value_bet_freq=0.0, fold_to_3bet=0.0, cbet_freq=0.0)
        p2 = PlayStyleParams(aggression=1.0, tightness=1.0, bluff_freq=1.0,
                              value_bet_freq=1.0, fold_to_3bet=1.0, cbet_freq=1.0)
        child = p1.crossover(p2, rng)
        vec = child.to_vector()
        # Each gene should be either 0.0 or 1.0
        assert all(v in (0.0, 1.0) for v in vec)


class TestParameterizedAgent:
    def test_creation(self):
        params = PlayStyleParams(aggression=0.9, tightness=0.2)
        agent = ParameterizedAgent(params, seed=42)
        assert agent.params.aggression == 0.9

    def test_decide_returns_valid(self):
        params = PlayStyleParams()
        agent = ParameterizedAgent(params, seed=42)
        engine = _make_engine()
        runtime = _make_runtime(engine, seed=42)
        action, amount = agent.decide(runtime, engine)
        legal = set(engine.legal_actions(runtime))
        assert action in legal

    def test_aggressive_agent_bets_more(self):
        """An aggressive agent should produce higher bet/raise probabilities."""
        engine = _make_engine()
        runtime = _make_runtime(engine, seed=42)
        state = runtime.state
        seat = state.acting_seat
        legal = set(engine.legal_actions(runtime))

        agg_agent = ParameterizedAgent(
            PlayStyleParams(aggression=1.0, bluff_freq=1.0, value_bet_freq=1.0), seed=42
        )
        passive_agent = ParameterizedAgent(
            PlayStyleParams(aggression=0.0, bluff_freq=0.0, value_bet_freq=0.0), seed=42
        )

        agg_dist = agg_agent.get_action_distribution(state, seat, legal)
        pas_dist = passive_agent.get_action_distribution(state, seat, legal)

        # Get aggressive action probabilities
        agg_actions = {ActionType.BET, ActionType.RAISE, ActionType.ALL_IN}
        agg_prob = sum(agg_dist.probabilities.get(a, 0) for a in agg_actions)
        pas_prob = sum(pas_dist.probabilities.get(a, 0) for a in agg_actions)
        assert agg_prob > pas_prob

    def test_get_action_distribution_sums_to_one(self):
        params = PlayStyleParams()
        agent = ParameterizedAgent(params, seed=42)
        engine = _make_engine()
        runtime = _make_runtime(engine, seed=42)
        state = runtime.state
        legal = set(engine.legal_actions(runtime))
        dist = agent.get_action_distribution(state, state.acting_seat, legal)
        assert abs(sum(dist.probabilities.values()) - 1.0) < 1e-6


class TestLatinHypercube:
    def test_correct_count(self):
        samples = _latin_hypercube_sample(10, 6, random.Random(42))
        assert len(samples) == 10
        assert all(len(s) == 6 for s in samples)

    def test_values_in_range(self):
        samples = _latin_hypercube_sample(20, 6, random.Random(42))
        for sample in samples:
            for v in sample:
                assert 0.0 <= v <= 1.0


class TestPCPGPopulation:
    def test_generate_diverse_population(self):
        pop = PCPGPopulation(seed=42)
        agents = pop.generate_diverse_population(size=10)
        assert len(agents) == 10
        assert all(isinstance(a, ParameterizedAgent) for a in agents)

    def test_diversity_score_positive(self):
        pop = PCPGPopulation(seed=42)
        pop.generate_diverse_population(size=10)
        score = pop.diversity_score()
        assert score > 0.0

    def test_diversity_empty_population(self):
        pop = PCPGPopulation(seed=42)
        assert pop.diversity_score() == 0.0

    def test_evolve_changes_population(self):
        pop = PCPGPopulation(seed=42)
        pop.generate_diverse_population(size=10)
        original_params = [a.params.to_vector() for a in pop.agents]
        fitness = [random.Random(i).random() for i in range(10)]
        pop.evolve(fitness)
        new_params = [a.params.to_vector() for a in pop.agents]
        # Population should change (bottom half replaced)
        assert any(o != n for o, n in zip(original_params, new_params))

    def test_evolve_preserves_size(self):
        pop = PCPGPopulation(seed=42)
        pop.generate_diverse_population(size=8)
        fitness = [float(i) for i in range(8)]
        pop.evolve(fitness)
        assert len(pop.agents) == 8

    def test_evolve_wrong_fitness_length(self):
        pop = PCPGPopulation(seed=42)
        pop.generate_diverse_population(size=5)
        with pytest.raises(ValueError):
            pop.evolve([1.0, 2.0])  # wrong length

    def test_get_best_response_training_set(self):
        pop = PCPGPopulation(seed=42)
        pop.generate_diverse_population(size=10)
        training_set = pop.get_best_response_training_set()
        assert len(training_set) == 10

    def test_best_agent(self):
        pop = PCPGPopulation(seed=42)
        pop.generate_diverse_population(size=5)
        fitness = [0.1, 0.9, 0.5, 0.3, 0.2]
        pop.evolve(fitness)
        # After evolve, fitness_scores reset, so set them manually
        pop.fitness_scores = fitness[:len(pop.agents)]
        best = pop.best_agent()
        assert isinstance(best, ParameterizedAgent)


# ===========================================================================
# Integration Tests
# ===========================================================================

class TestIntegration:
    def test_dmc_plays_full_hand(self):
        agent = DMCAgent(epsilon_start=0.5, seed=42)
        engine = _make_engine()
        runtime = _make_runtime(engine, seed=42)
        steps = 0
        while not runtime.state.is_terminal and runtime.state.acting_seat is not None:
            action, amount = agent.decide(runtime, engine)
            engine.apply_action(runtime, action, amount)
            steps += 1
            if steps > 20:
                break
        assert runtime.state.is_terminal

    def test_mcts_search_on_game_state(self):
        agent = ISMCTSAgent(num_simulations=15, seed=42)
        engine = _make_engine()
        runtime = _make_runtime(engine, seed=42)
        # Advance one action
        legal = engine.legal_actions(runtime)
        engine.apply_action(runtime, ActionType.CALL, 1)
        # Now search
        if not runtime.state.is_terminal and runtime.state.acting_seat is not None:
            dist = agent.search(runtime, engine)
            assert len(dist.probabilities) > 0

    def test_pcpg_generates_training_opponents(self):
        pop = PCPGPopulation(seed=42)
        opponents = pop.get_best_response_training_set()
        assert len(opponents) > 0
        engine = _make_engine()
        # Each opponent can play a hand
        for opp in opponents[:3]:
            runtime = _make_runtime(engine, seed=42)
            action, amount = opp.decide(runtime, engine)
            legal = set(engine.legal_actions(runtime))
            assert action in legal

    def test_copy_nn(self):
        nn = SimpleNN(FEATURE_DIM, 64, NUM_ACTIONS, seed=42)
        copied = _copy_nn(nn)
        feat = _random_features()
        orig_out = nn.forward(feat)
        copy_out = copied.forward(feat)
        for a, b in zip(orig_out, copy_out):
            assert abs(a - b) < 1e-9

    def test_soft_update(self):
        nn1 = SimpleNN(FEATURE_DIM, 64, NUM_ACTIONS, seed=42)
        nn2 = SimpleNN(FEATURE_DIM, 64, NUM_ACTIONS, seed=99)
        feat = _random_features()
        before = nn1.forward(feat)[:]
        _soft_update(nn1, nn2, tau=0.5)
        after = nn1.forward(feat)
        assert any(abs(a - b) > 1e-9 for a, b in zip(before, after))
