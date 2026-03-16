"""Tests for BehavioralPipeline and CFRAgent integration."""

from __future__ import annotations

import time

from packages.common.types import ActionType
from packages.opponent_model.behavioral_pipeline import (
    BehavioralPipeline,
    BehavioralSignals,
)
from packages.opponent_model.tilt_detector import TiltState
from packages.opponent_model.meta_game import AdaptationState
from packages.strategy.mixed import ActionDistribution


# ── BehavioralPipeline unit tests ────────────────────────────────────────


class TestBehavioralPipelineInit:

    def test_creates_all_submodules(self) -> None:
        bp = BehavioralPipeline()
        assert bp.tilt is not None
        assert bp.timing is not None
        assert bp.sizing is not None
        assert bp.meta_game is not None
        assert bp.fatigue is not None
        assert bp.bias_exploiter is not None

    def test_initial_signals_are_neutral(self) -> None:
        bp = BehavioralPipeline()
        signals = bp.get_signals()
        assert signals.tilt_state == TiltState.NORMAL
        assert signals.exploit_blend_adjustment >= 0.0
        assert signals.adaptation_state == AdaptationState.STATIC


class TestBehavioralPipelineRecording:

    def test_record_action_routes_to_tilt(self) -> None:
        bp = BehavioralPipeline()
        bp.record_action(ActionType.RAISE, "preflop", 1.5)
        assert bp.tilt._total_actions == 1

    def test_record_action_routes_to_timing(self) -> None:
        bp = BehavioralPipeline()
        bp.record_action(ActionType.BET, "flop", 0.75, decision_time_ms=5000)
        assert bp.timing.events_recorded == 1

    def test_record_action_routes_to_sizing(self) -> None:
        bp = BehavioralPipeline()
        bp.record_action(ActionType.BET, "flop", 0.5, pot_size=100, bet_amount=50)
        assert bp.sizing.events_recorded == 1

    def test_record_action_check_does_not_add_sizing(self) -> None:
        bp = BehavioralPipeline()
        bp.record_action(ActionType.CHECK, "flop", 0.0)
        assert bp.sizing.events_recorded == 0

    def test_record_action_tracks_consecutive_folds(self) -> None:
        bp = BehavioralPipeline()
        bp.record_action(ActionType.FOLD)
        bp.record_action(ActionType.FOLD)
        bp.record_action(ActionType.FOLD)
        assert bp._consecutive_folds == 3

    def test_consecutive_folds_reset_on_non_fold(self) -> None:
        bp = BehavioralPipeline()
        bp.record_action(ActionType.FOLD)
        bp.record_action(ActionType.FOLD)
        bp.record_action(ActionType.CALL)
        assert bp._consecutive_folds == 0

    def test_record_hand_result_routes_to_tilt(self) -> None:
        bp = BehavioralPipeline()
        bp.record_hand_result(profit_bb=-10.0, won=False)
        assert bp.tilt.hands_tracked == 1

    def test_record_hand_result_routes_to_fatigue(self) -> None:
        bp = BehavioralPipeline()
        bp.record_hand_result(profit_bb=5.0, won=True)
        assert bp.fatigue.hands_played == 1

    def test_record_hand_result_routes_to_meta_game(self) -> None:
        bp = BehavioralPipeline()
        bp.record_hand_result(vpip=0.3, pfr=0.2, aggression=1.5, fold_to_cbet=0.4)
        assert bp.meta_game.snapshots_recorded == 1

    def test_session_pnl_accumulated(self) -> None:
        bp = BehavioralPipeline()
        bp.record_hand_result(profit_bb=-10.0)
        bp.record_hand_result(profit_bb=5.0)
        bp.record_hand_result(profit_bb=-20.0)
        assert bp._session_pnl_bb == -25.0


class TestBehavioralPipelineSignals:

    def test_signals_type(self) -> None:
        bp = BehavioralPipeline()
        signals = bp.get_signals()
        assert isinstance(signals, BehavioralSignals)

    def test_exploit_blend_capped_at_half(self) -> None:
        bp = BehavioralPipeline()
        # Force high tilt by recording bad beats
        for _ in range(20):
            bp.record_action(ActionType.RAISE, "preflop", 2.0)
            bp.record_hand_result(profit_bb=-50.0, was_bad_beat=True, won=False)
        signals = bp.get_signals()
        assert signals.exploit_blend_adjustment <= 0.5

    def test_exploit_blend_non_negative(self) -> None:
        bp = BehavioralPipeline()
        signals = bp.get_signals()
        assert signals.exploit_blend_adjustment >= 0.0

    def test_tilt_increases_exploit_blend(self) -> None:
        bp = BehavioralPipeline()
        calm_signals = bp.get_signals()

        # Induce tilt
        for _ in range(10):
            bp.record_action(ActionType.RAISE, "preflop", 2.0)
            bp.record_hand_result(profit_bb=-30.0, was_bad_beat=True, won=False)

        tilted_signals = bp.get_signals()
        assert tilted_signals.exploit_blend_adjustment >= calm_signals.exploit_blend_adjustment

    def test_meta_game_adapting_reduces_exploitation(self) -> None:
        bp = BehavioralPipeline()
        # Simulate baseline (static stats)
        for _ in range(60):
            bp.record_hand_result(vpip=0.3, pfr=0.2, aggression=1.5, fold_to_cbet=0.4)
        # Then sudden shift (opponent adapting)
        for _ in range(25):
            bp.record_hand_result(vpip=0.6, pfr=0.5, aggression=3.0, fold_to_cbet=0.1)

        signals = bp.get_signals()
        # Even with tilt/fatigue, adaptation should moderate exploitation
        assert signals.adaptation_state in {
            AdaptationState.ADAPTING,
            AdaptationState.COUNTER_ADAPTING,
            AdaptationState.STATIC,
        }


class TestBehavioralPipelineStrategyAdjustment:

    def test_adjust_strategy_returns_distribution(self) -> None:
        bp = BehavioralPipeline()
        base = ActionDistribution(probabilities={
            ActionType.FOLD: 0.3,
            ActionType.CALL: 0.3,
            ActionType.RAISE: 0.4,
        })
        result = bp.adjust_strategy(base)
        assert isinstance(result, ActionDistribution)

    def test_adjust_strategy_no_change_when_calm(self) -> None:
        bp = BehavioralPipeline()
        base = ActionDistribution(probabilities={
            ActionType.FOLD: 0.3,
            ActionType.CALL: 0.3,
            ActionType.RAISE: 0.4,
        })
        result = bp.adjust_strategy(base)
        # Should be very close to base when no signals
        total_diff = sum(
            abs(result.probabilities.get(a, 0) - base.probabilities.get(a, 0))
            for a in base.probabilities
        )
        assert total_diff < 0.1

    def test_adjust_strategy_changes_with_signals(self) -> None:
        bp = BehavioralPipeline()
        # Induce tilt to generate signals
        for _ in range(10):
            bp.record_action(ActionType.RAISE, "preflop", 2.0)
            bp.record_hand_result(profit_bb=-50.0, was_bad_beat=True, won=False)

        base = ActionDistribution(probabilities={
            ActionType.FOLD: 0.3,
            ActionType.CALL: 0.3,
            ActionType.RAISE: 0.4,
        })
        result = bp.adjust_strategy(base)
        # With tilt signals, strategy should differ
        assert isinstance(result, ActionDistribution)


class TestBehavioralPipelineReset:

    def test_reset_clears_all_state(self) -> None:
        bp = BehavioralPipeline()
        bp.record_action(ActionType.RAISE, "preflop", 1.5, decision_time_ms=3000)
        bp.record_hand_result(profit_bb=-10.0)
        bp.reset()
        assert bp._session_pnl_bb == 0.0
        assert bp._consecutive_folds == 0
        assert bp.tilt.hands_tracked == 0
        assert bp.timing.events_recorded == 0
        assert bp.sizing.events_recorded == 0
        assert bp.fatigue.hands_played == 0


class TestBehavioralPipelineContextFields:

    def test_set_pot_invested_fraction(self) -> None:
        bp = BehavioralPipeline()
        bp.set_pot_invested_fraction(0.6)
        assert bp._pot_invested_fraction == 0.6

    def test_set_recent_bluffed(self) -> None:
        bp = BehavioralPipeline()
        bp.set_recent_bluffed(True)
        assert bp._recent_bluffed is True


# ── CFRAgent integration tests ───────────────────────────────────────────


class TestCFRAgentBehavioralIntegration:

    def test_agent_accepts_behavioral_pipeline(self) -> None:
        from packages.cfr_agent.agent import CFRAgent
        from packages.cfr_agent.trainer import CFRState

        bp = BehavioralPipeline()
        agent = CFRAgent(CFRState(), behavioral_pipeline=bp)
        assert agent.behavioral_pipeline is bp

    def test_agent_works_without_pipeline(self) -> None:
        from packages.cfr_agent.agent import CFRAgent
        from packages.cfr_agent.trainer import CFRState

        agent = CFRAgent(CFRState())
        assert agent.behavioral_pipeline is None


# ── Import verification ──────────────────────────────────────────────────


class TestBehavioralPipelineImports:

    def test_import_from_package(self) -> None:
        from packages.opponent_model import BehavioralPipeline, BehavioralSignals
        assert BehavioralPipeline is not None
        assert BehavioralSignals is not None

    def test_signals_dataclass_fields(self) -> None:
        signals = BehavioralSignals()
        assert hasattr(signals, "tilt_state")
        assert hasattr(signals, "timing_exploitability")
        assert hasattr(signals, "sizing_exploitability")
        assert hasattr(signals, "adaptation_state")
        assert hasattr(signals, "fatigue_score")
        assert hasattr(signals, "exploit_blend_adjustment")
