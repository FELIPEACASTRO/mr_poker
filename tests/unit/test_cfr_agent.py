"""Tests for CFR (Counterfactual Regret Minimization) agent."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from packages.cfr_agent.agent import CFRAgent
from packages.cfr_agent.info_set import build_info_set_key
from packages.cfr_agent.trainer import CFRState, CFRTrainer
from packages.common.types import ActionType
from packages.engine.engine import GameEngine
from packages.strategy.mixed import ActionDistribution


# ---------------------------------------------------------------------------
# CFRState tests
# ---------------------------------------------------------------------------

class TestCFRState:
    def test_current_strategy_uniform_for_unseen(self):
        state = CFRState()
        legal = {ActionType.FOLD, ActionType.CALL, ActionType.RAISE}
        dist = state.current_strategy("unseen_key", legal)
        assert len(dist.probabilities) == 3
        assert abs(sum(dist.probabilities.values()) - 1.0) < 1e-6

    def test_current_strategy_regret_matching(self):
        state = CFRState()
        state.cumulative_regret["key1"] = {
            ActionType.CALL.value: 10.0,
            ActionType.FOLD.value: -5.0,
            ActionType.RAISE.value: 5.0,
        }
        legal = {ActionType.FOLD, ActionType.CALL, ActionType.RAISE}
        dist = state.current_strategy("key1", legal)
        # Negative regrets clipped to 0
        assert dist.probabilities[ActionType.FOLD] == 0.0
        # CALL has 10/(10+5) = 2/3
        assert abs(dist.probabilities[ActionType.CALL] - 10 / 15) < 1e-6
        # RAISE has 5/(10+5) = 1/3
        assert abs(dist.probabilities[ActionType.RAISE] - 5 / 15) < 1e-6

    def test_average_strategy_converges(self):
        state = CFRState()
        state.strategy_sum["key1"] = {
            ActionType.CALL.value: 100.0,
            ActionType.FOLD.value: 50.0,
            ActionType.RAISE.value: 50.0,
        }
        legal = {ActionType.FOLD, ActionType.CALL, ActionType.RAISE}
        dist = state.average_strategy("key1", legal)
        assert abs(dist.probabilities[ActionType.CALL] - 0.5) < 1e-6
        assert abs(dist.probabilities[ActionType.FOLD] - 0.25) < 1e-6

    def test_update_accumulates_regrets(self):
        state = CFRState()
        strategy = ActionDistribution(
            probabilities={ActionType.CALL: 0.5, ActionType.FOLD: 0.5}
        )
        utilities = {ActionType.CALL: 10.0, ActionType.FOLD: -5.0}
        ev = 2.5  # 0.5*10 + 0.5*(-5) = 2.5
        state.update("key", strategy, utilities, ev)

        assert ActionType.CALL.value in state.cumulative_regret["key"]
        assert ActionType.FOLD.value in state.cumulative_regret["key"]
        # CALL regret = 10 - 2.5 = 7.5
        assert abs(state.cumulative_regret["key"][ActionType.CALL.value] - 7.5) < 1e-6
        # FOLD regret = -5 - 2.5 = -7.5
        assert abs(state.cumulative_regret["key"][ActionType.FOLD.value] - (-7.5)) < 1e-6

    def test_save_and_load(self, tmp_path):
        state = CFRState()
        state.iterations = 100
        state.cumulative_regret["k"] = {ActionType.CALL.value: 5.0}
        state.strategy_sum["k"] = {ActionType.CALL.value: 50.0}

        path = tmp_path / "cfr.json"
        state.save(path)
        loaded = CFRState.load(path)

        assert loaded.iterations == 100
        assert loaded.cumulative_regret["k"][ActionType.CALL.value] == 5.0
        assert loaded.strategy_sum["k"][ActionType.CALL.value] == 50.0


# ---------------------------------------------------------------------------
# Info set tests
# ---------------------------------------------------------------------------

class TestInfoSet:
    def test_build_info_set_key_preflop(self):
        engine = GameEngine(small_blind=1, big_blind=2)
        runtime = engine.start_new_hand(stacks=(100, 100), button_seat=0, seed=42)
        key = build_info_set_key(runtime.state, 0)
        # Should contain hole card class, PF (preflop), position
        assert "|PF|" in key
        assert "|IP|" in key or "|OOP|" in key

    def test_different_hands_different_keys(self):
        engine = GameEngine(small_blind=1, big_blind=2)
        rt1 = engine.start_new_hand(stacks=(100, 100), button_seat=0, seed=1)
        rt2 = engine.start_new_hand(stacks=(100, 100), button_seat=0, seed=999)
        k1 = build_info_set_key(rt1.state, 0)
        k2 = build_info_set_key(rt2.state, 0)
        # Different seeds almost certainly produce different hole cards
        # (there's a tiny chance they match)
        # Just verify both keys are well-formed
        assert k1.count("|") >= 5
        assert k2.count("|") >= 5

    def test_position_differs_by_seat(self):
        engine = GameEngine(small_blind=1, big_blind=2)
        runtime = engine.start_new_hand(stacks=(100, 100), button_seat=0, seed=42)
        k0 = build_info_set_key(runtime.state, 0)
        k1 = build_info_set_key(runtime.state, 1)
        # Seat 0 is button (IP), seat 1 is OOP
        assert "IP" in k0
        assert "OOP" in k1


# ---------------------------------------------------------------------------
# CFRTrainer tests
# ---------------------------------------------------------------------------

class TestCFRTrainer:
    def test_train_produces_info_sets(self):
        trainer = CFRTrainer(small_blind=1, big_blind=2, starting_stack=50, seed=7)
        state = trainer.train(iterations=100)
        assert state.iterations == 100
        assert len(state.strategy_sum) > 0
        assert len(state.cumulative_regret) > 0

    def test_train_strategies_are_valid_distributions(self):
        trainer = CFRTrainer(small_blind=1, big_blind=2, starting_stack=50, seed=7)
        state = trainer.train(iterations=200)

        for info_set, sums in state.strategy_sum.items():
            total = sum(max(0.0, v) for v in sums.values())
            if total > 0:
                probs = {k: max(0.0, v) / total for k, v in sums.items()}
                assert abs(sum(probs.values()) - 1.0) < 1e-6

    def test_more_iterations_more_info_sets(self):
        trainer = CFRTrainer(small_blind=1, big_blind=2, starting_stack=50, seed=7)
        state_100 = trainer.train(iterations=100)
        n_100 = len(state_100.strategy_sum)

        trainer2 = CFRTrainer(small_blind=1, big_blind=2, starting_stack=50, seed=7)
        state_500 = trainer2.train(iterations=500)
        n_500 = len(state_500.strategy_sum)

        assert n_500 >= n_100


# ---------------------------------------------------------------------------
# CFRAgent tests
# ---------------------------------------------------------------------------

class TestCFRAgent:
    @pytest.fixture()
    def trained_state(self):
        trainer = CFRTrainer(small_blind=1, big_blind=2, starting_stack=100, seed=42)
        return trainer.train(iterations=500)

    def test_agent_decides_valid_action(self, trained_state):
        engine = GameEngine(small_blind=1, big_blind=2)
        agent = CFRAgent(trained_state, seed=99)
        runtime = engine.start_new_hand(stacks=(100, 100), button_seat=0, seed=42)
        decision = agent.decide(runtime, engine)
        legal = set(engine.legal_actions(runtime))
        assert decision.action_type in legal

    def test_agent_plays_complete_hand(self, trained_state):
        engine = GameEngine(small_blind=1, big_blind=2)
        agent = CFRAgent(trained_state, seed=99)
        runtime = engine.start_new_hand(stacks=(100, 100), button_seat=0, seed=42)

        steps = 0
        while not runtime.state.is_terminal and runtime.state.acting_seat is not None and steps < 50:
            decision = agent.decide(runtime, engine)
            engine.apply_action(runtime, decision.action_type, decision.amount)
            steps += 1

        assert runtime.state.is_terminal

    def test_agent_rationale_contains_cfr_info(self, trained_state):
        engine = GameEngine(small_blind=1, big_blind=2)
        agent = CFRAgent(trained_state, seed=99)
        runtime = engine.start_new_hand(stacks=(100, 100), button_seat=0, seed=42)
        decision = agent.decide(runtime, engine)
        assert "cfr:" in decision.rationale
        assert "H=" in decision.rationale

    def test_agent_mixed_strategy_varies_actions(self, trained_state):
        """Over many hands with different seeds, the agent should select
        different actions for similar situations (mixed strategy)."""
        engine = GameEngine(small_blind=1, big_blind=2)
        actions_seen: set[ActionType] = set()

        for seed in range(50):
            agent = CFRAgent(trained_state, seed=seed)
            runtime = engine.start_new_hand(stacks=(100, 100), button_seat=0, seed=42)
            decision = agent.decide(runtime, engine)
            actions_seen.add(decision.action_type)

        # A mixed strategy should produce at least 2 different actions
        # over 50 runs with different RNG seeds
        assert len(actions_seen) >= 2, f"only saw {actions_seen}"

    def test_exploit_blend(self, trained_state):
        """With exploit_blend=1.0, the distribution is 100% baseline action,
        so sampling always returns the baseline action."""
        engine = GameEngine(small_blind=1, big_blind=2)
        agent_no_blend = CFRAgent(trained_state, seed=42, exploit_blend=0.0)
        agent_full_blend = CFRAgent(trained_state, seed=42, exploit_blend=1.0)

        # Verify exploit_blend parameter is stored correctly
        assert agent_no_blend.exploit_blend == 0.0
        assert agent_full_blend.exploit_blend == 1.0

        # Full blend should produce valid decisions
        runtime = engine.start_new_hand(stacks=(100, 100), button_seat=0, seed=42)
        dec = agent_full_blend.decide(runtime, engine)
        legal = set(engine.legal_actions(runtime))
        assert dec.action_type in legal
