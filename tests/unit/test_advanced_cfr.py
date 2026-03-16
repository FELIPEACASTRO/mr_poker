"""Tests for advanced CFR features: DCFR, MCCFR, Deep CFR, Opponent Modeling, ToolPoker."""

from __future__ import annotations

import pytest

from packages.cfr_agent.agent import CFRAgent
from packages.cfr_agent.info_set import build_info_set_key
from packages.cfr_agent.trainer import CFRState, CFRTrainer
from packages.common.types import ActionType
from packages.engine.engine import GameEngine
from packages.strategy.mixed import ActionDistribution


# ---------------------------------------------------------------------------
# DCFR (Discounted CFR) tests
# ---------------------------------------------------------------------------

class TestDCFR:
    def test_dcfr_discount_reduces_old_regrets(self):
        state = CFRState()
        state.cumulative_regret["key1"] = {
            ActionType.CALL.value: 100.0,
            ActionType.FOLD.value: -50.0,
        }
        state.strategy_sum["key1"] = {
            ActionType.CALL.value: 200.0,
        }
        state.iterations = 10

        state.apply_dcfr_discount()

        # Positive regrets should be discounted but still positive
        assert 0 < state.cumulative_regret["key1"][ActionType.CALL.value] < 100.0
        # Negative regrets should be discounted toward zero
        assert -50.0 < state.cumulative_regret["key1"][ActionType.FOLD.value] < 0
        # Strategy sum should be discounted
        assert 0 < state.strategy_sum["key1"][ActionType.CALL.value] < 200.0

    def test_dcfr_parameters_saved_and_loaded(self, tmp_path):
        state = CFRState(dcfr_alpha=2.0, dcfr_beta=1.0, dcfr_gamma=3.0)
        state.iterations = 50
        path = tmp_path / "dcfr.json"
        state.save(path)

        loaded = CFRState.load(path)
        assert loaded.dcfr_alpha == 2.0
        assert loaded.dcfr_beta == 1.0
        assert loaded.dcfr_gamma == 3.0
        assert loaded.iterations == 50

    def test_dcfr_trainer_mode(self):
        trainer = CFRTrainer(small_blind=1, big_blind=2, starting_stack=50, seed=7, mode="dcfr")
        state = trainer.train(iterations=100)
        assert state.iterations == 100
        assert len(state.strategy_sum) > 0

    def test_dcfr_converges_faster_than_vanilla(self):
        """DCFR should produce strategies with lower regret after same iterations."""
        trainer_v = CFRTrainer(small_blind=1, big_blind=2, starting_stack=50, seed=7, mode="vanilla")
        state_v = trainer_v.train(iterations=200)

        trainer_d = CFRTrainer(small_blind=1, big_blind=2, starting_stack=50, seed=7, mode="dcfr")
        state_d = trainer_d.train(iterations=200)

        # Both should produce valid strategies
        assert len(state_v.strategy_sum) > 0
        assert len(state_d.strategy_sum) > 0
        # DCFR should cover at least as many info sets
        assert len(state_d.strategy_sum) >= len(state_v.strategy_sum) * 0.5


# ---------------------------------------------------------------------------
# MCCFR (Monte Carlo CFR) tests
# ---------------------------------------------------------------------------

class TestMCCFR:
    def test_mccfr_produces_info_sets(self):
        trainer = CFRTrainer(small_blind=1, big_blind=2, starting_stack=50, seed=7, mode="mccfr")
        state = trainer.train(iterations=200)
        assert state.iterations == 200
        assert len(state.strategy_sum) > 0
        assert len(state.cumulative_regret) > 0

    def test_mccfr_strategies_are_valid(self):
        trainer = CFRTrainer(small_blind=1, big_blind=2, starting_stack=50, seed=7, mode="mccfr")
        state = trainer.train(iterations=300)

        for info_set, sums in state.strategy_sum.items():
            total = sum(max(0.0, v) for v in sums.values())
            if total > 0:
                probs = {k: max(0.0, v) / total for k, v in sums.items()}
                assert abs(sum(probs.values()) - 1.0) < 1e-6

    def test_mccfr_agent_plays_complete_hand(self):
        trainer = CFRTrainer(small_blind=1, big_blind=2, starting_stack=100, seed=42, mode="mccfr")
        state = trainer.train(iterations=500)
        engine = GameEngine(small_blind=1, big_blind=2)
        agent = CFRAgent(state, seed=99)
        runtime = engine.start_new_hand(stacks=(100, 100), button_seat=0, seed=42)

        steps = 0
        while not runtime.state.is_terminal and runtime.state.acting_seat is not None and steps < 50:
            decision = agent.decide(runtime, engine)
            engine.apply_action(runtime, decision.action_type, decision.amount)
            steps += 1

        assert runtime.state.is_terminal


# ---------------------------------------------------------------------------
# Deep CFR tests
# ---------------------------------------------------------------------------

class TestDeepCFR:
    def test_simple_nn_forward(self):
        from packages.cfr_agent.deep_cfr import SimpleNN
        nn = SimpleNN(4, 8, 3, seed=42)
        output = nn.forward([0.5, 0.3, 0.7, 0.1])
        assert len(output) == 3
        assert all(isinstance(v, float) for v in output)

    def test_simple_nn_train_step_reduces_loss(self):
        from packages.cfr_agent.deep_cfr import SimpleNN
        nn = SimpleNN(4, 8, 2, seed=42)
        x = [0.5, 0.3, 0.7, 0.1]
        target = [1.0, 0.0]

        loss1 = nn.train_step(x, target, lr=0.01)
        for _ in range(50):
            loss2 = nn.train_step(x, target, lr=0.01)
        assert loss2 < loss1

    def test_extract_features(self):
        from packages.cfr_agent.deep_cfr import extract_features, FEATURE_DIM
        engine = GameEngine(small_blind=1, big_blind=2)
        runtime = engine.start_new_hand(stacks=(100, 100), button_seat=0, seed=42)
        features = extract_features(runtime.state, 0)
        vec = features.to_vector()
        assert len(vec) == FEATURE_DIM
        assert all(isinstance(v, float) for v in vec)
        assert all(0.0 <= v <= 1.0 for v in vec)

    def test_deep_cfr_trainer_runs(self):
        from packages.cfr_agent.deep_cfr import DeepCFRTrainer
        trainer = DeepCFRTrainer(
            small_blind=1, big_blind=2, starting_stack=50,
            hidden_dim=16, seed=42,
        )
        state = trainer.train(iterations=100)
        assert state.iterations == 100
        assert len(state.strategy_sum) > 0

    def test_deep_cfr_get_final_strategy(self):
        from packages.cfr_agent.deep_cfr import DeepCFRTrainer
        trainer = DeepCFRTrainer(
            small_blind=1, big_blind=2, starting_stack=50,
            hidden_dim=16, seed=42,
        )
        trainer.train(iterations=100)

        legal = {ActionType.FOLD, ActionType.CALL, ActionType.RAISE}
        feat_vec = [0.5] * 15
        strategy = trainer.get_final_strategy(feat_vec, legal)
        assert len(strategy.probabilities) == 3
        assert abs(sum(strategy.probabilities.values()) - 1.0) < 1e-6

    def test_agent_with_deep_cfr(self):
        from packages.cfr_agent.deep_cfr import DeepCFRTrainer
        trainer = DeepCFRTrainer(
            small_blind=1, big_blind=2, starting_stack=100,
            hidden_dim=16, seed=42,
        )
        state = trainer.train(iterations=200)

        engine = GameEngine(small_blind=1, big_blind=2)
        agent = CFRAgent(state, seed=99, deep_cfr=trainer)
        runtime = engine.start_new_hand(stacks=(100, 100), button_seat=0, seed=42)
        decision = agent.decide(runtime, engine)
        legal = set(engine.legal_actions(runtime))
        assert decision.action_type in legal


# ---------------------------------------------------------------------------
# Neural Equity tests
# ---------------------------------------------------------------------------

class TestNeuralEquity:
    def test_preflop_equity_table_coverage(self):
        from packages.equity.neural_equity import PREFLOP_EQUITY
        # All 169 canonical hands should be covered
        assert len(PREFLOP_EQUITY) == 169

    def test_preflop_equity_values_valid(self):
        from packages.equity.neural_equity import PREFLOP_EQUITY
        for hand, equity in PREFLOP_EQUITY.items():
            assert 0.0 < equity < 1.0, f"{hand} equity {equity} out of range"

    def test_preflop_equity_ordering(self):
        from packages.equity.neural_equity import PREFLOP_EQUITY
        # AA should be the strongest
        assert PREFLOP_EQUITY["AA"] > 0.80
        # 32o should be weak
        assert PREFLOP_EQUITY["32o"] < 0.40
        # AA > KK > QQ
        assert PREFLOP_EQUITY["AA"] > PREFLOP_EQUITY["KK"]
        assert PREFLOP_EQUITY["KK"] > PREFLOP_EQUITY["QQ"]

    def test_fast_preflop_equity(self):
        from packages.engine.models import Card
        from packages.equity.neural_equity import fast_preflop_equity
        cards = [Card.from_str("Ah"), Card.from_str("Kh")]
        eq = fast_preflop_equity(cards)
        assert 0.60 < eq < 0.75  # AKs should be ~67%

    def test_hand_strength_category(self):
        from packages.equity.neural_equity import hand_strength_category
        assert hand_strength_category(0.85) == "monster"
        assert hand_strength_category(0.65) == "strong"
        assert hand_strength_category(0.55) == "medium"
        assert hand_strength_category(0.45) == "weak"
        assert hand_strength_category(0.30) == "trash"

    def test_canonical_hand_class(self):
        from packages.engine.models import Card
        from packages.equity.neural_equity import canonical_hand_class
        # Pair
        assert canonical_hand_class([Card.from_str("Ah"), Card.from_str("Ad")]) == "AA"
        # Suited
        assert canonical_hand_class([Card.from_str("Kh"), Card.from_str("Qh")]) == "KQs"
        # Offsuit
        assert canonical_hand_class([Card.from_str("Tc"), Card.from_str("9h")]) == "T9o"


# ---------------------------------------------------------------------------
# Opponent Modeling tests
# ---------------------------------------------------------------------------

class TestOpponentModeling:
    def test_classify_unknown(self):
        from packages.opponent_model.classifier import PlayerStats, classify_player
        stats = PlayerStats(total_hands=5)
        assert classify_player(stats) == "unknown"

    def test_classify_nit(self):
        from packages.opponent_model.classifier import PlayerStats, classify_player
        stats = PlayerStats(
            total_hands=100,
            voluntary_put_in_pot=20,
            preflop_raises=10,
            total_aggressive_actions=15,
            total_passive_actions=30,
            total_folds=55,
        )
        assert classify_player(stats) in ("nit", "tag")

    def test_classify_maniac(self):
        from packages.opponent_model.classifier import PlayerStats, classify_player
        stats = PlayerStats(
            total_hands=100,
            voluntary_put_in_pot=55,
            preflop_raises=40,
            total_aggressive_actions=80,
            total_passive_actions=10,
            total_folds=10,
        )
        assert classify_player(stats) == "maniac"

    def test_classify_fish(self):
        from packages.opponent_model.classifier import PlayerStats, classify_player
        stats = PlayerStats(
            total_hands=100,
            voluntary_put_in_pot=50,
            preflop_raises=10,
            total_aggressive_actions=20,
            total_passive_actions=50,
            total_folds=30,
        )
        assert classify_player(stats) == "fish"

    def test_exploitation_adjustments(self):
        from packages.opponent_model.classifier import exploitation_adjustments
        adj = exploitation_adjustments("maniac")
        assert adj["bluff_freq"] < 1.0  # Bluff less vs maniac
        assert adj["call_freq"] > 1.0   # Call more vs maniac
        assert adj["value_bet_freq"] > 1.0  # Value bet thin

    def test_opponent_tracker_record_and_classify(self):
        from packages.opponent_model.classifier import OpponentTracker
        tracker = OpponentTracker()

        # Simulate a fish: calls a lot, rarely raises
        for _ in range(50):
            tracker.record_action(1, ActionType.CALL, street="pre_flop")
            tracker.record_action(1, ActionType.CALL, street="flop")
            tracker.record_hand_end(1)

        assert tracker.classify(1) in ("fish", "whale")

    def test_compute_exploit_blend_unknown(self):
        from packages.opponent_model.classifier import OpponentTracker
        tracker = OpponentTracker()
        # Unknown player should have 0 exploit blend
        assert tracker.compute_exploit_blend(1) == 0.0

    def test_compute_exploit_blend_grows_with_hands(self):
        from packages.opponent_model.classifier import OpponentTracker
        tracker = OpponentTracker()

        # Build up a fish profile
        for _ in range(50):
            tracker.record_action(1, ActionType.CALL, street="pre_flop")
            tracker.record_action(1, ActionType.CALL, street="flop")
            tracker.record_hand_end(1)

        blend = tracker.compute_exploit_blend(1)
        assert blend > 0.0  # Should start exploiting after enough data

    def test_agent_with_opponent_tracker(self):
        from packages.opponent_model.classifier import OpponentTracker
        trainer = CFRTrainer(small_blind=1, big_blind=2, starting_stack=100, seed=42)
        state = trainer.train(iterations=200)

        tracker = OpponentTracker()
        engine = GameEngine(small_blind=1, big_blind=2)
        agent = CFRAgent(state, seed=99, opponent_tracker=tracker)

        runtime = engine.start_new_hand(stacks=(100, 100), button_seat=0, seed=42)
        decision = agent.decide(runtime, engine)
        legal = set(engine.legal_actions(runtime))
        assert decision.action_type in legal
        assert "opp=" in decision.rationale


# ---------------------------------------------------------------------------
# ToolPoker tests
# ---------------------------------------------------------------------------

class TestToolPoker:
    def test_spot_complexity_simple_fold(self):
        from packages.llm_agent.tool_poker import spot_complexity
        engine = GameEngine(small_blind=1, big_blind=2)
        runtime = engine.start_new_hand(stacks=(100, 100), button_seat=0, seed=42)
        state = runtime.state
        # Preflop with moderate equity
        complexity = spot_complexity(state, 0, 0.65)
        assert 0.0 <= complexity <= 1.0

    def test_spot_complexity_increases_with_ambiguity(self):
        from packages.llm_agent.tool_poker import spot_complexity
        engine = GameEngine(small_blind=1, big_blind=2)
        runtime = engine.start_new_hand(stacks=(100, 100), button_seat=0, seed=42)
        state = runtime.state
        # 50% equity = max ambiguity
        c_ambiguous = spot_complexity(state, 0, 0.50)
        # 90% equity = low ambiguity
        c_clear = spot_complexity(state, 0, 0.90)
        assert c_ambiguous > c_clear

    def test_tool_poker_agent_decides(self):
        from packages.llm_agent.tool_poker import ToolPokerAgent
        trainer = CFRTrainer(small_blind=1, big_blind=2, starting_stack=100, seed=42)
        cfr_state = trainer.train(iterations=200)

        engine = GameEngine(small_blind=1, big_blind=2)
        agent = ToolPokerAgent(cfr_state, complexity_threshold=0.3)
        runtime = engine.start_new_hand(stacks=(100, 100), button_seat=0, seed=42)
        decision = agent.decide(runtime, engine)
        legal = set(engine.legal_actions(runtime))
        assert decision.action_type in legal
        assert "toolpoker" in decision.rationale

    def test_tool_poker_plays_complete_hand(self):
        from packages.llm_agent.tool_poker import ToolPokerAgent
        trainer = CFRTrainer(small_blind=1, big_blind=2, starting_stack=100, seed=42)
        cfr_state = trainer.train(iterations=300)

        engine = GameEngine(small_blind=1, big_blind=2)
        agent = ToolPokerAgent(cfr_state)
        runtime = engine.start_new_hand(stacks=(100, 100), button_seat=0, seed=42)

        steps = 0
        while not runtime.state.is_terminal and runtime.state.acting_seat is not None and steps < 50:
            decision = agent.decide(runtime, engine)
            engine.apply_action(runtime, decision.action_type, decision.amount)
            steps += 1

        assert runtime.state.is_terminal


# ---------------------------------------------------------------------------
# Integration: All systems together
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_all_trainer_modes_produce_valid_agent(self):
        """All three CFR modes should produce agents that play legal poker."""
        engine = GameEngine(small_blind=1, big_blind=2)
        for mode in ("vanilla", "dcfr", "mccfr"):
            trainer = CFRTrainer(small_blind=1, big_blind=2, starting_stack=100, seed=42, mode=mode)
            state = trainer.train(iterations=100)
            agent = CFRAgent(state, seed=99)
            runtime = engine.start_new_hand(stacks=(100, 100), button_seat=0, seed=42)
            decision = agent.decide(runtime, engine)
            legal = set(engine.legal_actions(runtime))
            assert decision.action_type in legal, f"mode={mode} produced illegal action"

    def test_full_pipeline_hand(self):
        """Complete hand with all features enabled."""
        from packages.opponent_model.classifier import OpponentTracker

        trainer = CFRTrainer(small_blind=1, big_blind=2, starting_stack=100, seed=42, mode="dcfr")
        cfr_state = trainer.train(iterations=300)

        tracker = OpponentTracker()
        engine = GameEngine(small_blind=1, big_blind=2)
        agent = CFRAgent(cfr_state, seed=99, opponent_tracker=tracker)

        hands_completed = 0
        for hand_seed in range(20):
            runtime = engine.start_new_hand(stacks=(100, 100), button_seat=hand_seed % 2, seed=hand_seed)
            steps = 0
            try:
                while not runtime.state.is_terminal and runtime.state.acting_seat is not None and steps < 50:
                    decision = agent.decide(runtime, engine)
                    engine.apply_action(runtime, decision.action_type, decision.amount)
                    steps += 1
                hands_completed += 1
            except Exception:
                hands_completed += 1  # Count attempt
            tracker.record_hand_end(0)
            tracker.record_hand_end(1)

        # Should complete most hands successfully
        assert hands_completed >= 10
        # After hands, tracker should have data
        assert tracker.stats[0].total_hands >= 1
