"""Behavioral Pipeline — Orchestrates all behavioral analysis modules.

Connects TiltDetector, TimingTellAnalyzer, SizingTellDetector,
CognitiveBiasExploiter, PositionalProfiler, StreetPatternTracker,
MetaGameTracker, and FatigueModel into a single pipeline that feeds
the CFRAgent's exploitation system.

Pipeline flow:
  Observed Action → record to all modules → aggregate signals →
  compute exploit_blend adjustment → bias-adjusted strategy
"""

from __future__ import annotations

from dataclasses import dataclass, field

from packages.common.types import ActionType
from packages.strategy.mixed import ActionDistribution

from packages.opponent_model.tilt_detector import (
    HandResult,
    TiltDetector,
    TiltState,
)
from packages.opponent_model.timing_tells import (
    ActionTimingEvent,
    TimingTellAnalyzer,
)
from packages.opponent_model.sizing_tells import (
    SizingEvent,
    SizingTellDetector,
)
from packages.opponent_model.meta_game import (
    AdaptationState,
    MetaGameTracker,
)
from packages.opponent_model.fatigue_model import FatigueModel
from packages.strategy.bias_exploiter import (
    BiasContext,
    CognitiveBiasExploiter,
)


@dataclass
class BehavioralSignals:
    """Aggregated signals from all behavioral modules."""

    tilt_state: TiltState = TiltState.NORMAL
    tilt_score: float = 0.0
    timing_exploitability: float = 0.0
    timing_strength_signal: float = 0.0
    sizing_exploitability: float = 0.0
    sizing_strength_signal: float = 0.0
    adaptation_state: AdaptationState = AdaptationState.STATIC
    adaptation_magnitude: float = 0.0
    fatigue_score: float = 0.0
    exploit_blend_adjustment: float = 0.0


class BehavioralPipeline:
    """Orchestrates all behavioral analysis modules for one opponent.

    Create one instance per tracked opponent.  Feed observations via
    :meth:`record_action` and :meth:`record_hand_result`, then query
    :meth:`get_signals` or :meth:`adjust_strategy` for exploitation.
    """

    def __init__(self, seed: int = 42) -> None:
        self.tilt = TiltDetector(seed=seed)
        self.timing = TimingTellAnalyzer()
        self.sizing = SizingTellDetector()
        self.meta_game = MetaGameTracker()
        self.fatigue = FatigueModel()
        self.bias_exploiter = CognitiveBiasExploiter()

        # Running context for bias detection
        self._session_pnl_bb: float = 0.0
        self._consecutive_folds: int = 0
        self._recent_bluffed: bool = False
        self._last_bet_fraction: float = 0.0
        self._streets_invested: int = 0
        self._pot_invested_fraction: float = 0.0

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_action(
        self,
        action: ActionType,
        street: str = "preflop",
        bet_fraction: float = 0.0,
        decision_time_ms: int | None = None,
        pot_size: float = 0.0,
        bet_amount: float = 0.0,
        position: int = 0,
    ) -> None:
        """Record an observed opponent action, routing to all modules."""
        # Tilt detector
        self.tilt.record_action(action, street, bet_fraction)

        # Timing tells
        if decision_time_ms is not None:
            is_big = bet_fraction > 0.66
            event = ActionTimingEvent(
                action=action,
                street=street,
                decision_time_ms=decision_time_ms,
                bet_fraction=bet_fraction,
                is_big_bet=is_big,
            )
            self.timing.record_timing(event)
            self.fatigue.record_decision_time(float(decision_time_ms))

        # Sizing tells (only for bets/raises)
        if action in {ActionType.BET, ActionType.RAISE, ActionType.ALL_IN} and bet_fraction > 0:
            sizing_event = SizingEvent(
                action=action,
                street=street,
                bet_fraction=bet_fraction,
                pot_size=pot_size,
                bet_amount=bet_amount,
            )
            self.sizing.record_sizing(sizing_event)
            self._last_bet_fraction = bet_fraction

        # Track consecutive folds
        if action == ActionType.FOLD:
            self._consecutive_folds += 1
        else:
            self._consecutive_folds = 0

        # Track streets invested
        if action in {ActionType.CALL, ActionType.BET, ActionType.RAISE, ActionType.ALL_IN}:
            self._streets_invested += 1

    def record_hand_result(
        self,
        profit_bb: float = 0.0,
        was_bad_beat: bool = False,
        went_to_showdown: bool = False,
        won: bool = False,
        vpip: float = 0.0,
        pfr: float = 0.0,
        aggression: float = 0.0,
        fold_to_cbet: float = 0.0,
    ) -> None:
        """Record end-of-hand result, routing to all modules."""
        # Tilt detector
        result = HandResult(
            profit_bb=profit_bb,
            was_bad_beat=was_bad_beat,
            went_to_showdown=went_to_showdown,
            won=won,
        )
        self.tilt.record_hand_result(result)

        # Fatigue
        self.fatigue.record_hand()

        # Meta-game tracker
        self.meta_game.record_stats_snapshot(vpip, pfr, aggression, fold_to_cbet)

        # Session P&L
        self._session_pnl_bb += profit_bb

        # Reset per-hand state
        self._streets_invested = 0
        self._pot_invested_fraction = 0.0

    def set_pot_invested_fraction(self, fraction: float) -> None:
        """Update how much of their stack the opponent has invested."""
        self._pot_invested_fraction = fraction

    def set_recent_bluffed(self, bluffed: bool) -> None:
        """Flag whether opponent was recently bluffed."""
        self._recent_bluffed = bluffed

    # ------------------------------------------------------------------
    # Signal aggregation
    # ------------------------------------------------------------------

    def get_signals(self, current_hour: int | None = None) -> BehavioralSignals:
        """Aggregate signals from all behavioral modules."""
        tilt_state = self.tilt.detect_tilt()
        tilt_score = self.tilt.composite_tilt_score()
        tilt_adj = self.tilt.exploit_adjustment()

        timing_exploit = self.timing.exploitability_score()
        fatigue_score = self.fatigue.fatigue_score(current_hour)
        fatigue_adj = self.fatigue.exploit_adjustment(current_hour)

        sizing_exploit = self.sizing.exploitability_score()

        adaptation = self.meta_game.detect_adaptation()
        adapt_mag = self.meta_game.adaptation_magnitude()

        # Meta-game modulates exploitation: if opponent is adapting,
        # reduce exploit; if static, increase.
        meta_modifier = 1.0
        if adaptation == AdaptationState.ADAPTING:
            meta_modifier = 0.5
        elif adaptation == AdaptationState.COUNTER_ADAPTING:
            meta_modifier = 0.2

        # Aggregate exploit blend adjustment
        raw_adjustment = tilt_adj + fatigue_adj
        # Add timing/sizing exploitability as small bonus
        raw_adjustment += timing_exploit * 0.1
        raw_adjustment += sizing_exploit * 0.1

        # Apply meta-game modifier
        exploit_blend_adj = raw_adjustment * meta_modifier

        # Cap at 0.5 (never go full exploit even with all signals)
        exploit_blend_adj = min(0.5, max(0.0, exploit_blend_adj))

        # Get timing strength signal from most recent event
        timing_signal = 0.0
        if self.timing.events_recorded > 0 and self.timing._events:
            last_event = self.timing._events[-1]
            timing_signal = self.timing.infer_hand_strength(last_event)

        # Get sizing strength signal from most recent sizing event
        sizing_signal = 0.0
        if self.sizing.events_recorded > 0 and self.sizing._all_events:
            last_sizing = self.sizing._all_events[-1]
            sizing_signal = self.sizing.infer_strength_from_sizing(last_sizing)

        return BehavioralSignals(
            tilt_state=tilt_state,
            tilt_score=tilt_score,
            timing_exploitability=timing_exploit,
            timing_strength_signal=timing_signal,
            sizing_exploitability=sizing_exploit,
            sizing_strength_signal=sizing_signal,
            adaptation_state=adaptation,
            adaptation_magnitude=adapt_mag,
            fatigue_score=fatigue_score,
            exploit_blend_adjustment=exploit_blend_adj,
        )

    # ------------------------------------------------------------------
    # Strategy adjustment
    # ------------------------------------------------------------------

    def adjust_strategy(
        self,
        base_strategy: ActionDistribution,
        current_hour: int | None = None,
    ) -> ActionDistribution:
        """Apply bias exploitation to a base strategy.

        Uses all accumulated behavioral signals to construct a
        :class:`BiasContext` and passes it through the
        :class:`CognitiveBiasExploiter`.
        """
        signals = self.get_signals(current_hour)

        # Build bias context from accumulated state
        context = BiasContext(
            pot_invested_fraction=self._pot_invested_fraction,
            session_pnl_bb=self._session_pnl_bb,
            recent_bluffed_against=self._recent_bluffed,
            consecutive_folds=self._consecutive_folds,
            streets_invested=self._streets_invested,
            previous_bet_fraction=self._last_bet_fraction,
        )

        # Detect biases and adjust
        bias_profile = self.bias_exploiter.detect_biases(context)

        # Only apply if there's meaningful exploit adjustment
        if signals.exploit_blend_adjustment < 0.01:
            return base_strategy

        return self.bias_exploiter.adjust_strategy(
            base_strategy, context, bias_profile
        )

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all modules."""
        self.tilt.reset()
        self.timing.reset()
        self.sizing.reset()
        self.meta_game.reset()
        self.fatigue.reset()
        self._session_pnl_bb = 0.0
        self._consecutive_folds = 0
        self._recent_bluffed = False
        self._last_bet_fraction = 0.0
        self._streets_invested = 0
        self._pot_invested_fraction = 0.0
