"""Timing Tell Analysis -- Decision speed as behavioral signal.

Scientific basis:
- Slepian et al. (Psych Science 2013): timing more reliable than facial tells
- FG 2018 FACS: predict folds 3 seconds in advance
- Libratus: exploited timing patterns between sessions

Key patterns:
- Snap-action (<2s) + big bet = strong hand (high confidence)
- Long delay (>10s) + big bet = possible bluff (deliberation)
- Consistent timing = disciplined/balanced player (less exploitable)
- High timing variance = emotional/reactive player
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from packages.common.types import ActionType


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ActionTimingEvent:
    """Action with timing information."""

    action: ActionType = ActionType.CHECK
    street: str = "preflop"
    decision_time_ms: int = 3000  # milliseconds
    bet_fraction: float = 0.0  # bet size as fraction of pot
    is_big_bet: bool = False  # > 0.66x pot


@dataclass
class TimingProfile:
    """Statistical profile of decision timing."""

    mean_ms: float = 3000.0
    std_ms: float = 1500.0
    count: int = 0
    min_ms: float = 0.0
    max_ms: float = 0.0


# ---------------------------------------------------------------------------
# Internal running-stats accumulator
# ---------------------------------------------------------------------------

class _RunningStats:
    """Welford's online algorithm for mean and variance."""

    __slots__ = ("_n", "_mean", "_m2", "_min", "_max")

    def __init__(self) -> None:
        self._n: int = 0
        self._mean: float = 0.0
        self._m2: float = 0.0
        self._min: float = float("inf")
        self._max: float = float("-inf")

    def push(self, x: float) -> None:
        self._n += 1
        delta = x - self._mean
        self._mean += delta / self._n
        delta2 = x - self._mean
        self._m2 += delta * delta2
        if x < self._min:
            self._min = x
        if x > self._max:
            self._max = x

    @property
    def count(self) -> int:
        return self._n

    @property
    def mean(self) -> float:
        return self._mean if self._n > 0 else 0.0

    @property
    def variance(self) -> float:
        if self._n < 2:
            return 0.0
        return self._m2 / (self._n - 1)

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    @property
    def min_val(self) -> float:
        return self._min if self._n > 0 else 0.0

    @property
    def max_val(self) -> float:
        return self._max if self._n > 0 else 0.0

    def to_profile(self) -> TimingProfile:
        return TimingProfile(
            mean_ms=self.mean,
            std_ms=self.std,
            count=self._n,
            min_ms=self.min_val,
            max_ms=self.max_val,
        )


# ---------------------------------------------------------------------------
# TimingTellAnalyzer
# ---------------------------------------------------------------------------

_MAX_EVENTS = 2000  # cap stored events to bound memory


class TimingTellAnalyzer:
    """Analyzes decision timing for behavioral signals.

    Maintains running statistics per action type and per street, as well
    as an overall profile.  The primary outputs are:

    - :meth:`infer_hand_strength` — directional signal from a single event
    - :meth:`exploitability_score` — how exploitable timing patterns are
    - :meth:`z_score` — how unusual a specific timing is for this player
    """

    def __init__(
        self,
        snap_threshold_ms: int = 2000,
        tank_threshold_ms: int = 10000,
    ) -> None:
        self._snap_ms = max(1, snap_threshold_ms)
        self._tank_ms = max(self._snap_ms + 1, tank_threshold_ms)

        # Running stats
        self._overall: _RunningStats = _RunningStats()
        self._per_action: dict[ActionType, _RunningStats] = {}
        self._per_street: dict[str, _RunningStats] = {}

        # Raw event buffer (bounded)
        self._events: deque[ActionTimingEvent] = deque(maxlen=_MAX_EVENTS)

        # Correlation tracking (timing vs bet size) — for exploitability
        self._timing_bet_pairs: deque[tuple[float, float]] = deque(maxlen=_MAX_EVENTS)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_timing(self, event: ActionTimingEvent) -> None:
        """Record a timed action.

        Updates per-action, per-street, and overall profiles.
        """
        t = float(event.decision_time_ms)
        self._events.append(event)
        self._overall.push(t)

        # Per action type
        if event.action not in self._per_action:
            self._per_action[event.action] = _RunningStats()
        self._per_action[event.action].push(t)

        # Per street
        if event.street not in self._per_street:
            self._per_street[event.street] = _RunningStats()
        self._per_street[event.street].push(t)

        # Track timing-vs-bet for correlation
        if event.bet_fraction > 0.0:
            self._timing_bet_pairs.append((t, event.bet_fraction))

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def z_score(self, action: ActionType, time_ms: int) -> float:
        """Z-score of timing relative to player's baseline for this action.

        Returns 0.0 if fewer than 2 observations exist for that action.
        """
        stats = self._per_action.get(action)
        if stats is None or stats.count < 2 or stats.std < 1.0:
            return 0.0
        return (float(time_ms) - stats.mean) / stats.std

    def is_snap_action(self, time_ms: int) -> bool:
        """True if decision time is below the snap threshold."""
        return time_ms <= self._snap_ms

    def is_tank_action(self, time_ms: int) -> bool:
        """True if decision time exceeds the tank threshold."""
        return time_ms >= self._tank_ms

    def timing_variance(self) -> float:
        """Overall timing variance in ms^2 (high = emotional/reactive)."""
        return self._overall.variance

    def infer_hand_strength(self, event: ActionTimingEvent) -> float:
        """Infer hand strength signal from timing.

        Returns a value from -1.0 to +1.0:
        - Positive = likely strong hand
        - Negative = likely bluff / weak
        - Zero = neutral / insufficient information

        Heuristics (from research):
        - Snap action + big bet  ->  strong  (+0.7)
        - Tank action + big bet  ->  bluff   (-0.5)
        - Snap fold              ->  genuine fold (slight +0.2 for honesty)
        - Tank then check/call   ->  marginal hand (-0.2)
        - Normal timing          ->  neutral (0.0)
        """
        t = event.decision_time_ms
        snap = self.is_snap_action(t)
        tank = self.is_tank_action(t)
        big = event.is_big_bet

        aggressive = event.action in {ActionType.BET, ActionType.RAISE, ActionType.ALL_IN}
        passive = event.action in {ActionType.CHECK, ActionType.CALL}
        is_fold = event.action == ActionType.FOLD

        # Start with z-score contribution (smaller effect)
        z = self.z_score(event.action, t)
        z_contrib = 0.0
        if abs(z) > 1.5:
            # Unusually fast for this action -> more confident
            # Unusually slow for this action -> more deliberation
            z_contrib = -0.15 * (z / abs(z))  # sign: fast = positive

        signal = 0.0

        if snap and big and aggressive:
            signal = 0.7
        elif snap and aggressive and not big:
            signal = 0.3
        elif tank and big and aggressive:
            signal = -0.5
        elif tank and aggressive and not big:
            signal = -0.2
        elif snap and is_fold:
            signal = 0.2  # genuine give-up
        elif tank and passive:
            signal = -0.2  # marginal hand deliberation
        elif tank and is_fold:
            signal = -0.1  # was considering a bluff
        # else: normal timing -> 0.0

        signal += z_contrib
        return max(-1.0, min(1.0, signal))

    def exploitability_score(self) -> float:
        """How exploitable is this player's timing?

        Returns 0.0 (balanced/unreadable) to 1.0 (very exploitable).

        Based on:
        1. Coefficient of variation (CV) of overall timing — high CV means
           the player's timing leaks information.
        2. Correlation between decision time and bet size — strong
           correlation means timing reveals hand strength.

        Both components are combined 50/50.
        """
        if self._overall.count < 5:
            return 0.0

        # Component 1: CV of overall timing
        cv = self._overall.std / max(1.0, self._overall.mean)
        # CV of ~0.3 is "normal"; higher is more exploitable
        cv_score = min(1.0, max(0.0, (cv - 0.2) / 0.8))

        # Component 2: |correlation| between timing and bet size
        corr = self._timing_bet_correlation()
        corr_score = min(1.0, abs(corr))

        return 0.5 * cv_score + 0.5 * corr_score

    def get_profile(self, action: ActionType | None = None) -> TimingProfile:
        """Get timing profile, optionally filtered by action type.

        If *action* is ``None``, returns the overall profile.
        """
        if action is None:
            return self._overall.to_profile()
        stats = self._per_action.get(action)
        if stats is None:
            return TimingProfile(mean_ms=0.0, std_ms=0.0, count=0, min_ms=0.0, max_ms=0.0)
        return stats.to_profile()

    # ------------------------------------------------------------------
    # Reset & properties
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all recorded data."""
        self._overall = _RunningStats()
        self._per_action.clear()
        self._per_street.clear()
        self._events.clear()
        self._timing_bet_pairs.clear()

    @property
    def events_recorded(self) -> int:
        """Total number of timing events recorded."""
        return self._overall.count

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _timing_bet_correlation(self) -> float:
        """Pearson correlation between decision time and bet size.

        Returns 0.0 if insufficient data.
        """
        pairs = list(self._timing_bet_pairs)
        n = len(pairs)
        if n < 5:
            return 0.0

        sum_t = 0.0
        sum_b = 0.0
        sum_tt = 0.0
        sum_bb = 0.0
        sum_tb = 0.0

        for t, b in pairs:
            sum_t += t
            sum_b += b
            sum_tt += t * t
            sum_bb += b * b
            sum_tb += t * b

        mean_t = sum_t / n
        mean_b = sum_b / n
        var_t = sum_tt / n - mean_t * mean_t
        var_b = sum_bb / n - mean_b * mean_b

        if var_t < 1e-12 or var_b < 1e-12:
            return 0.0

        cov = sum_tb / n - mean_t * mean_b
        return cov / math.sqrt(var_t * var_b)

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"TimingTellAnalyzer(events={self.events_recorded}, "
            f"mean={self._overall.mean:.0f}ms, "
            f"exploit={self.exploitability_score():.2f})"
        )
