"""Tests for Batch 2: Strategy Modules.

Tests Bayesian Range Estimation, Kelly Criterion, AIVAT, and SAD Profiler.
"""

import math
import pytest

from packages.common.types import ActionType


# ── Bayesian Range Estimation ────────────────────────────────────────

class TestBayesianRange:
    def test_import(self):
        from packages.opponent_model.bayesian_range import (
            BayesianRangeEstimator, HAND_CLASSES, TOTAL_COMBOS,
        )
        assert len(HAND_CLASSES) == 169
        assert TOTAL_COMBOS == 1326

    def test_uniform_prior(self):
        from packages.opponent_model.bayesian_range import BayesianRangeEstimator
        est = BayesianRangeEstimator()
        post = est.posterior()
        total = sum(post.values())
        assert abs(total - 1.0) < 1e-6

    def test_update_narrows_range(self):
        from packages.opponent_model.bayesian_range import BayesianRangeEstimator
        est = BayesianRangeEstimator()
        initial_entropy = est.entropy()

        est.update(ActionType.RAISE, "preflop")
        est.update(ActionType.RAISE, "preflop")

        updated_entropy = est.entropy()
        assert updated_entropy < initial_entropy

    def test_raise_increases_premium_probability(self):
        from packages.opponent_model.bayesian_range import BayesianRangeEstimator
        est = BayesianRangeEstimator()
        post_before = est.posterior()
        aa_before = post_before.get("AA", 0)

        est.update(ActionType.RAISE, "preflop")
        est.update(ActionType.RAISE, "preflop")

        post_after = est.posterior()
        aa_after = post_after.get("AA", 0)
        assert aa_after > aa_before

    def test_fold_decreases_premium_probability(self):
        from packages.opponent_model.bayesian_range import BayesianRangeEstimator
        est = BayesianRangeEstimator()
        post_before = est.posterior()
        aa_before = post_before.get("AA", 0)

        est.update(ActionType.FOLD, "preflop")

        post_after = est.posterior()
        aa_after = post_after.get("AA", 0)
        assert aa_after < aa_before

    def test_map_estimate(self):
        from packages.opponent_model.bayesian_range import BayesianRangeEstimator
        est = BayesianRangeEstimator()
        est.update(ActionType.RAISE, "preflop")
        top = est.map_estimate(top_k=5)
        assert len(top) == 5
        assert all(isinstance(h, str) and isinstance(p, float) for h, p in top)
        # Should be sorted by probability
        probs = [p for _, p in top]
        assert probs == sorted(probs, reverse=True)

    def test_thompson_sample(self):
        from packages.opponent_model.bayesian_range import BayesianRangeEstimator, HAND_CLASSES
        est = BayesianRangeEstimator(seed=42)
        sample = est.thompson_sample()
        assert sample in HAND_CLASSES

    def test_range_equity_estimate(self):
        from packages.opponent_model.bayesian_range import BayesianRangeEstimator
        est = BayesianRangeEstimator()
        # After several raises, range should be strong
        for _ in range(5):
            est.update(ActionType.RAISE, "preflop")
        strength = est.range_equity_estimate()
        assert 0.0 <= strength <= 1.0

    def test_confidence(self):
        from packages.opponent_model.bayesian_range import BayesianRangeEstimator
        est = BayesianRangeEstimator()
        initial_conf = est.confidence()

        for _ in range(10):
            est.update(ActionType.RAISE, "preflop")

        final_conf = est.confidence()
        assert final_conf > initial_conf

    def test_reset(self):
        from packages.opponent_model.bayesian_range import BayesianRangeEstimator
        est = BayesianRangeEstimator()
        est.update(ActionType.RAISE, "preflop")
        est.reset()
        assert est._observations == 0

    def test_postflop_update(self):
        from packages.opponent_model.bayesian_range import BayesianRangeEstimator
        est = BayesianRangeEstimator()
        est.update(ActionType.BET, "postflop")
        post = est.posterior()
        total = sum(post.values())
        assert abs(total - 1.0) < 1e-6


# ── Kelly Criterion ──────────────────────────────────────────────────

class TestKellyCriterion:
    def test_import(self):
        from packages.strategy.kelly import (
            kelly_fraction, kelly_from_probability, half_kelly, quarter_kelly,
            BankrollManager,
        )
        assert kelly_fraction is not None

    def test_kelly_fraction_basic(self):
        from packages.strategy.kelly import kelly_fraction
        # Edge 0.1, odds 1.0 (even money) -> fraction = 0.1
        f = kelly_fraction(0.1, 1.0)
        assert abs(f - 0.1) < 1e-6

    def test_kelly_no_edge(self):
        from packages.strategy.kelly import kelly_fraction
        assert kelly_fraction(0.0, 1.0) == 0.0
        assert kelly_fraction(-0.1, 1.0) == 0.0

    def test_kelly_from_probability(self):
        from packages.strategy.kelly import kelly_from_probability
        # 60% win prob, even money: f* = (0.6 * 2 - 1) / 1 = 0.2
        f = kelly_from_probability(0.6, 1.0)
        assert abs(f - 0.2) < 1e-6

    def test_half_kelly(self):
        from packages.strategy.kelly import half_kelly, kelly_from_probability
        full = kelly_from_probability(0.6, 1.0)
        half = half_kelly(0.6, 1.0)
        assert abs(half - full * 0.5) < 1e-6

    def test_quarter_kelly(self):
        from packages.strategy.kelly import quarter_kelly, kelly_from_probability
        full = kelly_from_probability(0.6, 1.0)
        quarter = quarter_kelly(0.6, 1.0)
        assert abs(quarter - full * 0.25) < 1e-6

    def test_kelly_clamped(self):
        from packages.strategy.kelly import kelly_from_probability
        # Very high edge shouldn't exceed 1.0
        f = kelly_from_probability(0.99, 0.1)
        assert f <= 1.0

    def test_bankroll_manager(self):
        from packages.strategy.kelly import BankrollManager
        bm = BankrollManager(bankroll=10000.0, kelly_multiplier=0.5)
        bet = bm.recommended_bet(win_prob=0.6, pot_odds=1.0)
        assert bet > 0
        assert bet <= 10000.0 * 0.25  # max_bet_fraction

    def test_bankroll_update(self):
        from packages.strategy.kelly import BankrollManager
        bm = BankrollManager(bankroll=1000.0)
        bm.update_bankroll(100.0)
        assert bm.bankroll == 1100.0

    def test_risk_of_ruin(self):
        from packages.strategy.kelly import BankrollManager
        bm = BankrollManager(bankroll=5000.0)
        ror = bm.risk_of_ruin(edge=5.0, std_dev=60.0)
        assert 0.0 <= ror <= 1.0

    def test_buyins_remaining(self):
        from packages.strategy.kelly import BankrollManager
        bm = BankrollManager(bankroll=3000.0)
        assert bm.buyins_remaining(100.0) == 30.0

    def test_should_move_down(self):
        from packages.strategy.kelly import BankrollManager
        bm = BankrollManager(bankroll=1500.0)
        assert bm.should_move_down(100.0, min_buyins=20)  # 15 buyins < 20
        bm2 = BankrollManager(bankroll=3000.0)
        assert not bm2.should_move_down(100.0, min_buyins=20)  # 30 buyins >= 20

    def test_optimal_stake(self):
        from packages.strategy.kelly import BankrollManager
        bm = BankrollManager(bankroll=3000.0)
        assert bm.optimal_stake(30) == 100.0


# ── AIVAT ────────────────────────────────────────────────────────────

class TestAIVAT:
    def test_import(self):
        from packages.evaluation.aivat import AIVATEvaluator, AIVATResult, HandResult
        assert AIVATEvaluator is not None

    def test_empty_evaluation(self):
        from packages.evaluation.aivat import AIVATEvaluator
        ev = AIVATEvaluator()
        result = ev.evaluate()
        assert result.num_hands == 0
        assert result.raw_winrate_bb100 == 0.0

    def test_single_hand(self):
        from packages.evaluation.aivat import AIVATEvaluator, HandResult
        ev = AIVATEvaluator()
        ev.add_hand(HandResult(
            hand_id="1", hero_seat=0, raw_result_bb=5.0,
            decision_equities=[0.6, 0.7],
            decision_pots_bb=[4.0, 8.0],
            hero_acted=[False, False],
            final_equity=0.8,
        ))
        result = ev.evaluate()
        assert result.num_hands == 1
        assert result.raw_winrate_bb100 == 500.0

    def test_variance_reduction(self):
        from packages.evaluation.aivat import AIVATEvaluator, HandResult
        import random
        rng = random.Random(42)
        ev = AIVATEvaluator()

        for i in range(100):
            # Simulate hands with card luck variance
            equity = 0.5 + rng.gauss(0, 0.15)
            raw = 2.0 + (equity - 0.5) * 20.0 + rng.gauss(0, 5.0)
            ev.add_hand(HandResult(
                hand_id=str(i), hero_seat=0, raw_result_bb=raw,
                decision_equities=[equity],
                decision_pots_bb=[10.0],
                hero_acted=[False],
                final_equity=equity,
            ))

        result = ev.evaluate()
        assert result.num_hands == 100
        # AIVAT std should be lower than raw std
        assert result.aivat_std_dev <= result.raw_std_dev

    def test_confidence_interval(self):
        from packages.evaluation.aivat import AIVATEvaluator, HandResult
        import random
        rng = random.Random(42)
        ev = AIVATEvaluator()
        for i in range(50):
            ev.add_hand(HandResult(
                hand_id=str(i), hero_seat=0,
                raw_result_bb=1.0 + rng.gauss(0, 2.0),
                final_equity=0.5 + rng.gauss(0, 0.1),
            ))
        result = ev.evaluate()
        low, high = result.confidence_interval_95
        assert low < high

    def test_minimum_hands(self):
        from packages.evaluation.aivat import AIVATEvaluator
        ev = AIVATEvaluator()
        n = ev.minimum_hands_for_significance(expected_winrate_bb100=5.0)
        assert n > 0

    def test_reset(self):
        from packages.evaluation.aivat import AIVATEvaluator, HandResult
        ev = AIVATEvaluator()
        ev.add_hand(HandResult(hand_id="1", hero_seat=0, raw_result_bb=1.0))
        ev.reset()
        assert ev.evaluate().num_hands == 0

    def test_import_from_package(self):
        from packages.evaluation import AIVATEvaluator, AIVATResult, HandResult
        assert AIVATEvaluator is not None


# ── SAD Profiler ─────────────────────────────────────────────────────

class TestSADProfiler:
    def test_import(self):
        from packages.opponent_model.sad_profiler import (
            SADProfiler, SADProfile, Leak, LeakType,
        )
        assert SADProfiler is not None

    def _make_stats(self, **overrides) -> "PlayerStats":
        from packages.opponent_model.classifier import PlayerStats
        defaults = dict(
            total_hands=100,
            voluntary_put_in_pot=25,
            preflop_raises=20,
            three_bets=8,
            three_bet_opportunities=100,
            fold_to_three_bet=50,
            cbet_made=40,
            cbet_opportunities=60,
            fold_to_cbet=20,
            cbet_faced=50,
            total_aggressive_actions=50,
            total_passive_actions=30,
            total_folds=40,
            went_to_showdown=25,
            won_at_showdown=13,
            total_postflop_actions=80,
            flop_actions=30, flop_aggression=15,
            turn_actions=25, turn_aggression=10,
            river_actions=20, river_aggression=8,
        )
        defaults.update(overrides)
        return PlayerStats(**defaults)

    def test_analyze_standard_player(self):
        from packages.opponent_model.sad_profiler import SADProfiler
        profiler = SADProfiler()
        stats = self._make_stats()
        profile = profiler.analyze("player1", stats)
        assert profile.player_id == "player1"
        assert profile.hands_analyzed == 100

    def test_detect_folds_too_much(self):
        from packages.opponent_model.sad_profiler import SADProfiler, LeakType
        profiler = SADProfiler()
        stats = self._make_stats(
            total_folds=80,
            total_aggressive_actions=10,
            total_passive_actions=10,
        )
        profile = profiler.analyze("nit", stats)
        leak_types = [l.leak_type for l in profile.leaks]
        assert LeakType.FOLDS_TOO_MUCH in leak_types

    def test_detect_calls_too_much(self):
        from packages.opponent_model.sad_profiler import SADProfiler, LeakType
        profiler = SADProfiler()
        stats = self._make_stats(voluntary_put_in_pot=55)
        profile = profiler.analyze("fish", stats)
        leak_types = [l.leak_type for l in profile.leaks]
        assert LeakType.CALLS_TOO_MUCH in leak_types

    def test_detect_passive_postflop(self):
        from packages.opponent_model.sad_profiler import SADProfiler, LeakType
        profiler = SADProfiler()
        stats = self._make_stats(
            total_aggressive_actions=5,
            total_passive_actions=50,
            total_postflop_actions=55,
        )
        profile = profiler.analyze("passive", stats)
        leak_types = [l.leak_type for l in profile.leaks]
        assert LeakType.PASSIVE_POSTFLOP in leak_types

    def test_not_enough_hands(self):
        from packages.opponent_model.sad_profiler import SADProfiler
        profiler = SADProfiler(min_hands=50)
        stats = self._make_stats(total_hands=10)
        profile = profiler.analyze("new", stats)
        assert len(profile.leaks) == 0
        assert not profile.has_actionable_leaks

    def test_top_leaks_sorted(self):
        from packages.opponent_model.sad_profiler import SADProfiler
        profiler = SADProfiler()
        stats = self._make_stats(
            total_folds=80,
            total_aggressive_actions=5,
            total_passive_actions=15,
            total_postflop_actions=20,
        )
        profile = profiler.analyze("weak", stats)
        if len(profile.top_leaks) >= 2:
            assert profile.top_leaks[0].severity >= profile.top_leaks[1].severity

    def test_exploit_adjustments(self):
        from packages.opponent_model.sad_profiler import SADProfiler
        profiler = SADProfiler()
        stats = self._make_stats(
            total_folds=80,
            total_aggressive_actions=10,
            total_passive_actions=10,
        )
        profile = profiler.analyze("nit", stats)
        adjustments = profiler.generate_exploit_adjustments(profile)
        assert "bluff_frequency" in adjustments

    def test_leak_is_significant(self):
        from packages.opponent_model.sad_profiler import Leak, LeakType
        leak = Leak(
            leak_type=LeakType.FOLDS_TOO_MUCH,
            severity=0.5,
            description="test",
            counter_strategy="test",
            observed_value=0.7,
            expected_gto_range=(0.30, 0.45),
            sample_size=50,
        )
        assert leak.is_significant

    def test_leak_not_significant_small_sample(self):
        from packages.opponent_model.sad_profiler import Leak, LeakType
        leak = Leak(
            leak_type=LeakType.FOLDS_TOO_MUCH,
            severity=0.5,
            description="test",
            counter_strategy="test",
            observed_value=0.7,
            expected_gto_range=(0.30, 0.45),
            sample_size=5,
        )
        assert not leak.is_significant

    def test_import_from_package(self):
        from packages.opponent_model import SADProfiler, BayesianRangeEstimator
        assert SADProfiler is not None
        assert BayesianRangeEstimator is not None
