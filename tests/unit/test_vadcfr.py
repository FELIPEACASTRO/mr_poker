"""Tests for VAD-CFR (Volatility-Adaptive Discounted CFR) implementation."""

import math
import pytest

from packages.cfr_agent.trainer import CFRState, CFRTrainer
from packages.common.types import ActionType
from packages.strategy.mixed import ActionDistribution


# ---------------------------------------------------------------------------
# CFRState.update_vad tests
# ---------------------------------------------------------------------------

class TestUpdateVad:
    """Tests for the VAD-CFR update method on CFRState."""

    def _make_state(self, iterations: int = 0, warmup: int = 50) -> CFRState:
        s = CFRState()
        s.iterations = iterations
        s.vad_warmup = warmup
        return s

    def _legal(self):
        return {ActionType.FOLD, ActionType.CALL, ActionType.RAISE}

    def _uniform_strategy(self, legal):
        n = len(legal)
        return ActionDistribution(probabilities={a: 1.0 / n for a in legal})

    def test_basic_regret_accumulation(self):
        """update_vad accumulates regrets like vanilla CFR."""
        state = self._make_state(iterations=0)
        legal = self._legal()
        strategy = self._uniform_strategy(legal)
        utilities = {ActionType.FOLD: -1.0, ActionType.CALL: 0.5, ActionType.RAISE: 2.0}
        ev = sum(strategy.probabilities[a] * utilities[a] for a in legal)

        state.update_vad("IS1", strategy, utilities, ev)

        for a in legal:
            expected_regret = utilities[a] - ev
            actual = state.cumulative_regret["IS1"][a.value]
            # Should be close to expected (optimism bonus may add a tiny amount)
            assert abs(actual - expected_regret) < 0.5, f"{a}: {actual} vs {expected_regret}"

    def test_volatility_tracking_initialized(self):
        """update_vad creates volatility entries."""
        state = self._make_state()
        legal = self._legal()
        strategy = self._uniform_strategy(legal)
        utilities = {ActionType.FOLD: -1.0, ActionType.CALL: 0.0, ActionType.RAISE: 1.0}
        ev = 0.0

        state.update_vad("IS1", strategy, utilities, ev)

        assert "IS1" in state._vad_volatility
        for a in legal:
            assert a.value in state._vad_volatility["IS1"]
            prev_regret, vol = state._vad_volatility["IS1"][a.value]
            assert isinstance(prev_regret, float)
            assert isinstance(vol, float)

    def test_volatility_increases_with_oscillation(self):
        """Alternating regret signs should increase volatility."""
        state = self._make_state()
        legal = {ActionType.FOLD, ActionType.CALL}
        strategy = ActionDistribution(probabilities={ActionType.FOLD: 0.5, ActionType.CALL: 0.5})

        # Feed oscillating utilities
        for i in range(20):
            if i % 2 == 0:
                utils = {ActionType.FOLD: 1.0, ActionType.CALL: -1.0}
            else:
                utils = {ActionType.FOLD: -1.0, ActionType.CALL: 1.0}
            state.update_vad("IS_OSC", strategy, utils, 0.0)

        vol_fold = state._vad_volatility["IS_OSC"][ActionType.FOLD.value][1]
        assert vol_fold > 0.1, f"Volatility should be high for oscillating: {vol_fold}"

    def test_volatility_low_for_consistent_regrets(self):
        """Consistent regrets should keep volatility low."""
        state = self._make_state()
        legal = {ActionType.FOLD, ActionType.CALL}
        strategy = ActionDistribution(probabilities={ActionType.FOLD: 0.5, ActionType.CALL: 0.5})

        # Feed consistent utilities
        for _ in range(20):
            utils = {ActionType.FOLD: 1.0, ActionType.CALL: -1.0}
            state.update_vad("IS_CONST", strategy, utils, 0.0)

        vol_fold = state._vad_volatility["IS_CONST"][ActionType.FOLD.value][1]
        # After consistent input, volatility should be near 0
        assert vol_fold < 0.05, f"Volatility should be low for consistent: {vol_fold}"

    def test_no_strategy_accumulation_before_warmup(self):
        """Strategy sum should not accumulate before vad_warmup."""
        state = self._make_state(iterations=10, warmup=50)
        legal = self._legal()
        strategy = self._uniform_strategy(legal)
        utilities = {ActionType.FOLD: -1.0, ActionType.CALL: 0.5, ActionType.RAISE: 2.0}
        ev = 0.5

        state.update_vad("IS_WARM", strategy, utilities, ev)

        # Strategy sum should be empty (iterations=10 < warmup=50)
        strat_sum = state.strategy_sum.get("IS_WARM", {})
        total = sum(strat_sum.values())
        assert total == 0.0, f"Strategy sum should be 0 before warmup: {total}"

    def test_strategy_accumulation_after_warmup(self):
        """Strategy sum should accumulate after vad_warmup."""
        state = self._make_state(iterations=60, warmup=50)
        legal = self._legal()
        strategy = self._uniform_strategy(legal)
        utilities = {ActionType.FOLD: -1.0, ActionType.CALL: 0.5, ActionType.RAISE: 2.0}
        ev = 0.5

        state.update_vad("IS_POST", strategy, utilities, ev)

        strat_sum = state.strategy_sum.get("IS_POST", {})
        total = sum(strat_sum.values())
        assert total > 0.0, f"Strategy sum should accumulate after warmup: {total}"

    def test_optimism_bonus_for_consistent_positive_regret(self):
        """Consistently positive regret with low volatility gets optimism bonus."""
        state = self._make_state(iterations=0)
        legal = {ActionType.FOLD, ActionType.RAISE}
        strategy = ActionDistribution(probabilities={ActionType.FOLD: 0.5, ActionType.RAISE: 0.5})

        # Build up positive cumulative regret for RAISE with consistent signal
        for _ in range(30):
            utils = {ActionType.FOLD: -2.0, ActionType.RAISE: 2.0}
            state.update_vad("IS_OPT", strategy, utils, 0.0)

        # The cumulative regret for RAISE should be higher than just sum of regrets
        # due to optimism bonus
        pure_sum = 2.0 * 30  # 30 iterations * regret=2.0
        actual = state.cumulative_regret["IS_OPT"][ActionType.RAISE.value]
        # With optimism, should be >= pure sum
        assert actual >= pure_sum, f"Optimism should boost: {actual} vs {pure_sum}"


# ---------------------------------------------------------------------------
# CFRState.apply_vad_discount tests
# ---------------------------------------------------------------------------

class TestApplyVadDiscount:
    """Tests for volatility-adaptive discounting."""

    def test_discounts_positive_regrets(self):
        """Positive regrets should be discounted (retained partially)."""
        state = CFRState()
        state.iterations = 100
        state.cumulative_regret["IS1"] = {"fold": 10.0, "call": -5.0, "raise": 3.0}
        state._vad_volatility["IS1"] = {
            "fold": (1.0, 0.1),
            "call": (-1.0, 0.1),
            "raise": (1.0, 0.1),
        }

        state.apply_vad_discount()

        # Positive regrets should be discounted but still positive
        assert 0 < state.cumulative_regret["IS1"]["fold"] < 10.0
        assert 0 < state.cumulative_regret["IS1"]["raise"] < 3.0
        # Negative regrets should be discounted toward 0
        assert -5.0 < state.cumulative_regret["IS1"]["call"] < 0

    def test_high_volatility_stronger_discount(self):
        """Higher volatility should result in stronger discounting."""
        # Create two states: one with high vol, one with low vol
        state_high = CFRState()
        state_high.iterations = 50
        state_high.cumulative_regret["IS1"] = {"fold": 10.0}
        state_high._vad_volatility["IS1"] = {"fold": (1.0, 5.0)}  # high vol

        state_low = CFRState()
        state_low.iterations = 50
        state_low.cumulative_regret["IS1"] = {"fold": 10.0}
        state_low._vad_volatility["IS1"] = {"fold": (1.0, 0.01)}  # low vol

        state_high.apply_vad_discount()
        state_low.apply_vad_discount()

        # High volatility → more aggressive discount → lower remaining value
        assert state_high.cumulative_regret["IS1"]["fold"] <= state_low.cumulative_regret["IS1"]["fold"]

    def test_strategy_sum_not_discounted_before_warmup(self):
        """Strategy sums should not be discounted if iterations < warmup."""
        state = CFRState()
        state.iterations = 30
        state.vad_warmup = 50
        state.strategy_sum["IS1"] = {"fold": 5.0, "call": 3.0}
        state.cumulative_regret["IS1"] = {"fold": 1.0, "call": 1.0}

        state.apply_vad_discount()

        # Strategy sum unchanged (iterations < warmup)
        assert state.strategy_sum["IS1"]["fold"] == 5.0
        assert state.strategy_sum["IS1"]["call"] == 3.0

    def test_strategy_sum_discounted_after_warmup(self):
        """Strategy sums should be discounted after warmup."""
        state = CFRState()
        state.iterations = 100
        state.vad_warmup = 50
        state.strategy_sum["IS1"] = {"fold": 5.0, "call": 3.0}
        state.cumulative_regret["IS1"] = {"fold": 1.0, "call": 1.0}

        state.apply_vad_discount()

        assert state.strategy_sum["IS1"]["fold"] < 5.0
        assert state.strategy_sum["IS1"]["call"] < 3.0

    def test_handles_empty_volatility(self):
        """Should not crash when volatility data is missing for an info set."""
        state = CFRState()
        state.iterations = 50
        state.cumulative_regret["IS_NO_VOL"] = {"fold": 5.0}
        # No volatility data for IS_NO_VOL

        state.apply_vad_discount()  # Should not raise

        assert state.cumulative_regret["IS_NO_VOL"]["fold"] < 5.0


# ---------------------------------------------------------------------------
# CFRTrainer with mode="vadcfr" integration tests
# ---------------------------------------------------------------------------

class TestVadCfrTrainer:
    """Integration tests for VAD-CFR training mode."""

    def test_vadcfr_mode_runs(self):
        """VAD-CFR training should complete without errors."""
        trainer = CFRTrainer(
            small_blind=1,
            big_blind=2,
            starting_stack=100,
            seed=42,
            mode="vadcfr",
        )
        state = trainer.train(iterations=50)

        assert state.iterations == 50
        assert len(state.strategy_sum) > 0
        assert len(state.cumulative_regret) > 0

    def test_vadcfr_produces_valid_strategies(self):
        """VAD-CFR average strategies should be valid probability distributions."""
        trainer = CFRTrainer(
            small_blind=1,
            big_blind=2,
            starting_stack=100,
            seed=123,
            mode="vadcfr",
        )
        state = trainer.train(iterations=100)

        legal = {ActionType.FOLD, ActionType.CHECK, ActionType.CALL, ActionType.BET, ActionType.RAISE, ActionType.ALL_IN}
        for info_set in list(state.strategy_sum.keys())[:10]:
            strat = state.average_strategy(info_set, legal)
            total = sum(strat.probabilities.values())
            assert abs(total - 1.0) < 1e-6, f"Strategy for {info_set} doesn't sum to 1: {total}"
            for p in strat.probabilities.values():
                assert p >= 0.0, f"Negative probability in {info_set}"

    def test_vadcfr_tracks_volatility(self):
        """VAD-CFR should populate volatility tracking data."""
        trainer = CFRTrainer(
            small_blind=1,
            big_blind=2,
            starting_stack=100,
            seed=42,
            mode="vadcfr",
        )
        state = trainer.train(iterations=50)

        assert len(state._vad_volatility) > 0, "Should have volatility data"

    def test_vadcfr_warmup_delays_strategy(self):
        """With high warmup, early iterations shouldn't accumulate strategy."""
        trainer = CFRTrainer(
            small_blind=1,
            big_blind=2,
            starting_stack=100,
            seed=42,
            mode="vadcfr",
        )
        trainer.cfr_state.vad_warmup = 1000  # Very high warmup

        state = trainer.train(iterations=20)

        # All strategy sums should be 0 or empty
        for info_set, sums in state.strategy_sum.items():
            total = sum(sums.values())
            assert total == 0.0, f"Strategy sum should be 0 during warmup: {info_set}={total}"

    def test_vadcfr_vs_dcfr_both_produce_info_sets(self):
        """Both VAD-CFR and DCFR should discover info sets and accumulate data."""
        trainer_vad = CFRTrainer(small_blind=1, big_blind=2, starting_stack=100, seed=42, mode="vadcfr")
        trainer_vad.cfr_state.vad_warmup = 10  # Low warmup for short test
        state_vad = trainer_vad.train(iterations=200)

        trainer_dcfr = CFRTrainer(small_blind=1, big_blind=2, starting_stack=100, seed=42, mode="dcfr")
        state_dcfr = trainer_dcfr.train(iterations=200)

        # Both should discover info sets
        assert len(state_vad.strategy_sum) > 0, "VAD-CFR should discover info sets"
        assert len(state_dcfr.strategy_sum) > 0, "DCFR should discover info sets"
        assert len(state_vad.cumulative_regret) > 0
        assert len(state_dcfr.cumulative_regret) > 0

        # VAD-CFR should have volatility tracking that DCFR lacks
        assert len(state_vad._vad_volatility) > 0, "VAD-CFR should track volatility"
        assert len(state_dcfr._vad_volatility) == 0, "DCFR should not track volatility"

    def test_vadcfr_discount_applied_periodically(self):
        """VAD discount should fire every 100 iterations."""
        trainer = CFRTrainer(
            small_blind=1,
            big_blind=2,
            starting_stack=100,
            seed=42,
            mode="vadcfr",
        )
        trainer.cfr_state.vad_warmup = 10

        # Run 150 iterations — discount should fire at iteration 100
        state = trainer.train(iterations=150)

        # Verify it ran (we can't easily check discount was applied,
        # but we verify no crash and strategies exist)
        assert state.iterations == 150
        assert len(state.strategy_sum) > 0
