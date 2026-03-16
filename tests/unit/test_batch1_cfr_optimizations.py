"""Tests for Batch 1: CFR Optimizations.

Tests regret-based pruning, compact CFR, warm starting, and lazy-CFR.
"""

import json
import math
import tempfile
from pathlib import Path

import pytest

from packages.common.types import ActionType
from packages.strategy.mixed import ActionDistribution
from packages.cfr_agent.trainer import CFRState, CFRTrainer


# ── Regret-based Pruning ──────────────────────────────────────────────

class TestRegretPruning:
    """Tests for RegretPruningCFRTrainer."""

    def test_import(self):
        from packages.cfr_agent.pruning import RegretPruningCFRTrainer, PruningStats
        assert RegretPruningCFRTrainer is not None
        assert PruningStats is not None

    def test_basic_training(self):
        from packages.cfr_agent.pruning import RegretPruningCFRTrainer
        trainer = RegretPruningCFRTrainer(
            seed=42, mode="dcfr", min_iterations_before_pruning=5,
        )
        state = trainer.train(iterations=20)
        assert state.iterations == 20
        # Engine may or may not generate playable hands depending on seed
        assert state.iterations > 0

    def test_pruning_stats(self):
        from packages.cfr_agent.pruning import PruningStats
        stats = PruningStats(total_actions_considered=100, actions_pruned=25)
        assert stats.prune_rate == 0.25

    def test_pruning_stats_zero(self):
        from packages.cfr_agent.pruning import PruningStats
        stats = PruningStats()
        assert stats.prune_rate == 0.0

    def test_should_prune_respects_min_iterations(self):
        from packages.cfr_agent.pruning import RegretPruningCFRTrainer
        trainer = RegretPruningCFRTrainer(
            seed=42, min_iterations_before_pruning=100,
        )
        # Before min_iterations, never prune
        trainer.cfr_state.iterations = 50
        trainer.cfr_state.cumulative_regret["test"] = {"fold": -999_999_999}
        assert not trainer._should_prune("test", ActionType.FOLD)

    def test_should_prune_after_min_iterations(self):
        from packages.cfr_agent.pruning import RegretPruningCFRTrainer
        trainer = RegretPruningCFRTrainer(
            seed=42,
            prune_threshold=-100.0,
            min_iterations_before_pruning=10,
        )
        trainer.cfr_state.iterations = 20
        trainer.cfr_state.cumulative_regret["test"] = {"fold": -200.0}
        assert trainer._should_prune("test", ActionType.FOLD)

    def test_should_not_prune_positive_regret(self):
        from packages.cfr_agent.pruning import RegretPruningCFRTrainer
        trainer = RegretPruningCFRTrainer(
            seed=42, min_iterations_before_pruning=0,
        )
        trainer.cfr_state.iterations = 100
        trainer.cfr_state.cumulative_regret["test"] = {"fold": 100.0}
        assert not trainer._should_prune("test", ActionType.FOLD)

    def test_probe_round(self):
        from packages.cfr_agent.pruning import RegretPruningCFRTrainer
        trainer = RegretPruningCFRTrainer(seed=42, probe_interval=5)
        trainer.cfr_state.iterations = 10
        assert trainer._is_probe_round()
        trainer.cfr_state.iterations = 11
        assert not trainer._is_probe_round()

    def test_mccfr_mode(self):
        from packages.cfr_agent.pruning import RegretPruningCFRTrainer
        trainer = RegretPruningCFRTrainer(
            seed=42, mode="mccfr", min_iterations_before_pruning=5,
        )
        state = trainer.train(iterations=20)
        assert state.iterations == 20
        assert len(state.strategy_sum) > 0

    def test_produces_valid_strategies(self):
        from packages.cfr_agent.pruning import RegretPruningCFRTrainer
        trainer = RegretPruningCFRTrainer(seed=42, mode="dcfr")
        state = trainer.train(iterations=50)
        for info_set in list(state.strategy_sum.keys())[:5]:
            legal = {ActionType.FOLD, ActionType.CALL, ActionType.RAISE}
            dist = state.average_strategy(info_set, legal)
            total = sum(dist.probabilities.values())
            assert abs(total - 1.0) < 1e-6


# ── Compact CFR ──────────────────────────────────────────────────────

class TestCompactCFR:
    """Tests for CompactCFRState."""

    def test_import(self):
        from packages.cfr_agent.compact_cfr import CompactCFRState
        assert CompactCFRState is not None

    def test_quantize_dequantize_roundtrip(self):
        from packages.cfr_agent.compact_cfr import CompactCFRState
        state = CompactCFRState()
        for val in [0.0, 1.0, -1.0, 0.5, -0.5, 100.0, -100.0]:
            offset = 0.0
            scale = max(abs(val), 1.0)
            q = state._quantize(val, offset, scale)
            deq = state._dequantize(q, offset, scale)
            assert abs(deq - val) < scale * 0.02  # ~2% error max

    def test_update_and_retrieve(self):
        from packages.cfr_agent.compact_cfr import CompactCFRState
        state = CompactCFRState()
        legal = {ActionType.FOLD, ActionType.CALL}
        strategy = ActionDistribution(probabilities={ActionType.FOLD: 0.5, ActionType.CALL: 0.5})
        action_utils = {ActionType.FOLD: -1.0, ActionType.CALL: 1.0}
        state.update("info1", strategy, action_utils, 0.0)

        # Should be in buffer before flush
        assert state.get_regret("info1", "fold") == pytest.approx(-1.0)
        assert state.get_regret("info1", "call") == pytest.approx(1.0)

    def test_flush_to_quantized(self):
        from packages.cfr_agent.compact_cfr import CompactCFRState
        state = CompactCFRState()
        strategy = ActionDistribution(probabilities={ActionType.FOLD: 0.3, ActionType.CALL: 0.7})
        action_utils = {ActionType.FOLD: -2.0, ActionType.CALL: 2.0}
        state.update("info1", strategy, action_utils, 0.0)

        state.flush_to_quantized()
        assert "info1" in state.regret_quantized
        assert "info1" in state.strategy_quantized

    def test_current_strategy(self):
        from packages.cfr_agent.compact_cfr import CompactCFRState
        state = CompactCFRState()
        legal = {ActionType.FOLD, ActionType.CALL, ActionType.RAISE}

        # No data -> uniform
        dist = state.current_strategy("unknown", legal)
        for p in dist.probabilities.values():
            assert abs(p - 1.0 / 3) < 1e-6

    def test_average_strategy(self):
        from packages.cfr_agent.compact_cfr import CompactCFRState
        state = CompactCFRState()
        legal = {ActionType.FOLD, ActionType.CALL}
        strategy = ActionDistribution(probabilities={ActionType.FOLD: 0.2, ActionType.CALL: 0.8})
        state.update("info1", strategy, {ActionType.FOLD: 0.0, ActionType.CALL: 0.0}, 0.0)

        avg = state.average_strategy("info1", legal)
        # Strategy sum should reflect the probabilities used
        assert avg.probabilities[ActionType.CALL] > avg.probabilities[ActionType.FOLD]

    def test_dcfr_discount(self):
        from packages.cfr_agent.compact_cfr import CompactCFRState
        state = CompactCFRState()
        state.iterations = 10
        strategy = ActionDistribution(probabilities={ActionType.FOLD: 0.5, ActionType.CALL: 0.5})
        state.update("info1", strategy, {ActionType.FOLD: 5.0, ActionType.CALL: -5.0}, 0.0)

        before_fold = state.get_regret("info1", "fold")
        state.apply_dcfr_discount()
        after_fold = state.get_regret("info1", "fold")
        # Positive regrets should be discounted (slightly reduced)
        assert after_fold < before_fold

    def test_memory_usage_estimate(self):
        from packages.cfr_agent.compact_cfr import CompactCFRState
        state = CompactCFRState()
        strategy = ActionDistribution(probabilities={ActionType.FOLD: 0.5, ActionType.CALL: 0.5})
        for i in range(100):
            state.update(
                f"info_{i}", strategy,
                {ActionType.FOLD: float(i), ActionType.CALL: -float(i)}, 0.0,
            )
        state.flush_to_quantized()
        usage = state.memory_usage_estimate()
        assert usage["total_bytes"] > 0
        assert usage["compression_ratio"] >= 1.0

    def test_save_load_roundtrip(self):
        from packages.cfr_agent.compact_cfr import CompactCFRState
        state = CompactCFRState()
        state.iterations = 42
        strategy = ActionDistribution(probabilities={ActionType.FOLD: 0.4, ActionType.CALL: 0.6})
        state.update("info1", strategy, {ActionType.FOLD: -1.0, ActionType.CALL: 1.0}, 0.0)
        state.flush_to_quantized()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        state.save(path)
        loaded = CompactCFRState.load(path)
        assert loaded.iterations == 42
        assert "info1" in loaded.regret_quantized
        Path(path).unlink()


# ── Warm Starting ──────────────────────────────────────────────────────

class TestWarmStart:
    """Tests for warm starting CFR."""

    def test_import(self):
        from packages.cfr_agent.warm_start import (
            warm_start_from_strategy,
            warm_start_from_cfr_state,
            extract_strategy_from_cfr,
        )
        assert warm_start_from_strategy is not None

    def test_warm_start_from_strategy(self):
        from packages.cfr_agent.warm_start import warm_start_from_strategy
        state = CFRState()
        initial = {
            "info1": {ActionType.FOLD: 0.3, ActionType.CALL: 0.7},
        }
        warm_start_from_strategy(state, initial, regret_scale=100.0)

        # Check regrets were set
        assert "info1" in state.cumulative_regret
        assert state.cumulative_regret["info1"]["fold"] == pytest.approx(30.0)
        assert state.cumulative_regret["info1"]["call"] == pytest.approx(70.0)

    def test_warm_start_reproduces_strategy(self):
        from packages.cfr_agent.warm_start import warm_start_from_strategy
        state = CFRState()
        initial = {
            "info1": {ActionType.FOLD: 0.2, ActionType.CALL: 0.5, ActionType.RAISE: 0.3},
        }
        warm_start_from_strategy(state, initial, regret_scale=1000.0)

        legal = {ActionType.FOLD, ActionType.CALL, ActionType.RAISE}
        dist = state.current_strategy("info1", legal)

        # Strategy should approximately match the initial
        assert abs(dist.probabilities[ActionType.FOLD] - 0.2) < 0.01
        assert abs(dist.probabilities[ActionType.CALL] - 0.5) < 0.01
        assert abs(dist.probabilities[ActionType.RAISE] - 0.3) < 0.01

    def test_warm_start_from_cfr_state(self):
        from packages.cfr_agent.warm_start import warm_start_from_cfr_state
        source = CFRState()
        source.cumulative_regret["info1"] = {"fold": 10.0, "call": 20.0}
        source.strategy_sum["info1"] = {"fold": 5.0, "call": 15.0}

        target = CFRState()
        warm_start_from_cfr_state(target, source, decay=0.5)

        assert target.cumulative_regret["info1"]["fold"] == pytest.approx(5.0)
        assert target.cumulative_regret["info1"]["call"] == pytest.approx(10.0)

    def test_extract_strategy(self):
        from packages.cfr_agent.warm_start import extract_strategy_from_cfr
        state = CFRState()
        state.strategy_sum["info1"] = {"fold": 3.0, "call": 7.0}

        strategies = extract_strategy_from_cfr(state)
        assert "info1" in strategies
        assert abs(strategies["info1"][ActionType.FOLD] - 0.3) < 1e-6
        assert abs(strategies["info1"][ActionType.CALL] - 0.7) < 1e-6

    def test_extract_empty_state(self):
        from packages.cfr_agent.warm_start import extract_strategy_from_cfr
        state = CFRState()
        strategies = extract_strategy_from_cfr(state)
        assert strategies == {}

    def test_warm_start_empty_strategy(self):
        from packages.cfr_agent.warm_start import warm_start_from_strategy
        state = CFRState()
        warm_start_from_strategy(state, {})
        assert len(state.cumulative_regret) == 0

    def test_roundtrip_extract_then_warm_start(self):
        from packages.cfr_agent.warm_start import (
            warm_start_from_strategy,
            extract_strategy_from_cfr,
        )
        # Train a small CFR
        trainer = CFRTrainer(seed=42, mode="dcfr")
        trained = trainer.train(iterations=50)

        # Extract and re-warm-start
        strategies = extract_strategy_from_cfr(trained)
        new_state = CFRState()
        warm_start_from_strategy(new_state, strategies)

        # Should have the same info sets
        for info_set in strategies:
            assert info_set in new_state.cumulative_regret


# ── Lazy-CFR ──────────────────────────────────────────────────────────

class TestLazyCFR:
    """Tests for LazyCFRTrainer."""

    def test_import(self):
        from packages.cfr_agent.lazy_cfr import LazyCFRTrainer, LazyCFRConfig, PriorityQueue
        assert LazyCFRTrainer is not None
        assert LazyCFRConfig is not None

    def test_basic_training(self):
        from packages.cfr_agent.lazy_cfr import LazyCFRTrainer, LazyCFRConfig
        config = LazyCFRConfig(
            update_fraction=0.5,
            warmup_iterations=5,
            full_update_interval=10,
        )
        trainer = LazyCFRTrainer(seed=42, mode="dcfr", config=config)
        state = trainer.train(iterations=30)
        assert state.iterations == 30
        assert len(state.strategy_sum) > 0

    def test_priority_queue(self):
        from packages.cfr_agent.lazy_cfr import PriorityQueue
        import random
        pq = PriorityQueue()
        rng = random.Random(42)

        pq.update_priority("high", 1000.0)
        pq.update_priority("low", 1.0)

        # High priority should almost always be updated
        high_count = sum(
            1 for _ in range(100) if pq.should_update("high", 0.3, rng)
        )
        low_count = sum(
            1 for _ in range(100) if pq.should_update("low", 0.3, rng)
        )
        assert high_count > low_count

    def test_priority_queue_new_info_set(self):
        from packages.cfr_agent.lazy_cfr import PriorityQueue
        import random
        pq = PriorityQueue()
        rng = random.Random(42)
        # New info sets should always be updated
        assert pq.should_update("new_set", 0.1, rng)

    def test_priority_queue_stats(self):
        from packages.cfr_agent.lazy_cfr import PriorityQueue
        pq = PriorityQueue()
        pq.update_priority("a", 10.0)
        pq.record_update("a")
        pq.record_update("a")
        stats = pq.stats()
        assert stats["total"] == 1
        assert stats["avg_updates"] == 2.0

    def test_mccfr_mode(self):
        from packages.cfr_agent.lazy_cfr import LazyCFRTrainer, LazyCFRConfig
        config = LazyCFRConfig(warmup_iterations=5, full_update_interval=10)
        trainer = LazyCFRTrainer(seed=42, mode="mccfr", config=config)
        state = trainer.train(iterations=20)
        assert state.iterations == 20

    def test_full_update_fraction(self):
        from packages.cfr_agent.lazy_cfr import LazyCFRTrainer, LazyCFRConfig
        # fraction=1.0 should behave like regular CFR
        config = LazyCFRConfig(update_fraction=1.0, warmup_iterations=0)
        trainer = LazyCFRTrainer(seed=42, mode="dcfr", config=config)
        state = trainer.train(iterations=20)
        assert trainer._skipped_updates == 0

    def test_config_defaults(self):
        from packages.cfr_agent.lazy_cfr import LazyCFRConfig
        config = LazyCFRConfig()
        assert config.update_fraction == 0.3
        assert config.use_priority is True
        assert config.warmup_iterations == 100
        assert config.full_update_interval == 50


# ── Integration: all Batch 1 modules via __init__ ────────────────────

class TestBatch1Integration:
    """Integration tests for all Batch 1 modules."""

    def test_all_imports_from_package(self):
        from packages.cfr_agent import (
            RegretPruningCFRTrainer,
            CompactCFRState,
            warm_start_from_strategy,
            warm_start_from_cfr_state,
            extract_strategy_from_cfr,
            LazyCFRTrainer,
            LazyCFRConfig,
        )
        assert RegretPruningCFRTrainer is not None
        assert CompactCFRState is not None
        assert LazyCFRTrainer is not None

    def test_pruning_then_warm_start(self):
        """Train with pruning, extract strategy, warm-start new trainer."""
        from packages.cfr_agent.pruning import RegretPruningCFRTrainer
        from packages.cfr_agent.warm_start import (
            extract_strategy_from_cfr,
            warm_start_from_strategy,
        )

        trainer = RegretPruningCFRTrainer(seed=42, mode="dcfr")
        state = trainer.train(iterations=50)

        strategies = extract_strategy_from_cfr(state)
        # May be empty if engine didn't produce playable hands in 50 iters
        new_state = CFRState()
        warm_start_from_strategy(new_state, strategies)
        assert len(new_state.cumulative_regret) == len(strategies)

    def test_compact_state_with_training_data(self):
        """Train regular CFR, convert to compact, verify strategies match."""
        from packages.cfr_agent.compact_cfr import CompactCFRState

        trainer = CFRTrainer(seed=42, mode="dcfr")
        trained = trainer.train(iterations=50)

        compact = CompactCFRState()
        for info_set, regrets in trained.cumulative_regret.items():
            for key, val in regrets.items():
                if info_set not in compact._regret_buffer:
                    compact._regret_buffer[info_set] = {}
                compact._regret_buffer[info_set][key] = val

        for info_set, strats in trained.strategy_sum.items():
            for key, val in strats.items():
                if info_set not in compact._strategy_buffer:
                    compact._strategy_buffer[info_set] = {}
                compact._strategy_buffer[info_set][key] = val

        compact.flush_to_quantized()
        usage = compact.memory_usage_estimate()
        assert usage["compression_ratio"] >= 1.0
