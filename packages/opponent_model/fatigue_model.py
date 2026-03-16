"""Fatigue and Session Duration Modeling.

Models the likelihood that an opponent is fatigued based on:
- Session duration (minutes played)
- Time of day (circadian rhythm)
- Decision speed trends (slowing down = fatiguing)

Scientific basis:
- Csikszentmihalyi Flow State: optimal performance window 30-90 min.
- J. Sleep Research: sleep deprivation reduces math performance by 30%,
  increases impulsivity and risk-taking.
- Poker-specific: tired players widen ranges, make larger sizing errors,
  and tilt more easily after bad beats.

When fatigue is detected, we increase exploitation blend because the
opponent is more likely to deviate from GTO in predictable ways:
- Wider preflop ranges (calling too much)
- Less disciplined bet sizing
- Increased tilt susceptibility
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field


class FatigueLevel(enum.Enum):
    """Discrete fatigue levels."""

    FRESH = "fresh"          # < 30 min
    NORMAL = "normal"        # 30-90 min
    TIRED = "tired"          # 90-180 min
    EXHAUSTED = "exhausted"  # > 180 min


@dataclass
class SessionInfo:
    """Session timing information."""

    start_time: float = 0.0  # Unix timestamp
    hands_played: int = 0
    current_hour: int = 12   # 0-23 hour of day


# Circadian penalty: hours where cognitive performance is lowest
_NIGHT_HOURS = set(range(0, 7))  # midnight to 6 AM


class FatigueModel:
    """Models opponent fatigue based on session duration and time of day.

    Usage::

        model = FatigueModel()
        # ... during session ...
        model.record_hand()
        model.record_decision_time(2300.0)  # ms
        level = model.estimate_fatigue()
        extra_exploit = model.exploit_adjustment()
    """

    def __init__(self) -> None:
        self.session_start: float = time.time()
        self.hands_played: int = 0
        self._decision_times: list[float] = []
        self._max_decision_history = 200

    # -- recording ---------------------------------------------------------

    def record_hand(self) -> None:
        """Record that a hand has been completed."""
        self.hands_played += 1

    def record_decision_time(self, time_ms: float) -> None:
        """Track decision speed in milliseconds for slowdown detection.

        Only the most recent ``_max_decision_history`` times are kept.
        """
        self._decision_times.append(time_ms)
        if len(self._decision_times) > self._max_decision_history:
            self._decision_times = self._decision_times[-self._max_decision_history:]

    # -- queries -----------------------------------------------------------

    def session_duration_minutes(self) -> float:
        """Minutes elapsed since session start."""
        return (time.time() - self.session_start) / 60.0

    def estimate_fatigue(self, current_hour: int | None = None) -> FatigueLevel:
        """Estimate fatigue from duration + time of day.

        Night hours (0-6) bump the fatigue level up by one tier.
        """
        minutes = self.session_duration_minutes()
        base_level = self._duration_to_level(minutes)

        if current_hour is not None and current_hour in _NIGHT_HOURS:
            base_level = self._bump_level(base_level)

        return base_level

    def fatigue_score(self, current_hour: int | None = None) -> float:
        """Continuous fatigue score: 0.0 (fresh) to 1.0 (exhausted).

        Combines session duration, time-of-day penalty, and decision
        speed trend.
        """
        minutes = self.session_duration_minutes()

        # Duration component: 0 at 0 min, ~0.5 at 90 min, ~0.8 at 180 min
        # Using a saturating curve: 1 - exp(-minutes / 180)
        import math
        duration_score = 1.0 - math.exp(-minutes / 180.0)

        # Circadian component
        circadian_penalty = 0.0
        if current_hour is not None and current_hour in _NIGHT_HOURS:
            # Strongest penalty at 3-4 AM
            if 2 <= current_hour <= 5:
                circadian_penalty = 0.20
            else:
                circadian_penalty = 0.10

        # Decision speed component
        speed_penalty = 0.0
        trend = self.decision_speed_trend()
        if trend > 0:
            # Slowing down: each 100ms/hand increase adds penalty
            speed_penalty = min(0.15, trend / 1000.0)

        raw = duration_score + circadian_penalty + speed_penalty
        return min(1.0, max(0.0, raw))

    def exploit_adjustment(self, current_hour: int | None = None) -> float:
        """Additional exploitation blend based on fatigue.

        Returns 0.0 to 0.2.  Add this to the base exploit blend from
        the opponent classifier.
        """
        score = self.fatigue_score(current_hour)
        # Linear mapping: 0.0 fatigue -> 0.0 adjustment, 1.0 -> 0.20
        return score * 0.20

    def decision_speed_trend(self) -> float:
        """Trend in decision speed (slope of linear regression).

        >0 = slowing down (fatiguing).
        <0 = speeding up.
        0.0 = stable or insufficient data.

        Units: milliseconds per hand.
        """
        times = self._decision_times
        n = len(times)
        if n < 10:
            return 0.0

        # Use the last 50 data points (or all if fewer)
        window = times[-50:]
        m = len(window)

        # Simple linear regression: y = a + b*x
        # x = index (0, 1, ..., m-1), y = decision time
        sum_x = 0.0
        sum_y = 0.0
        sum_xy = 0.0
        sum_x2 = 0.0

        for i, y in enumerate(window):
            x = float(i)
            sum_x += x
            sum_y += y
            sum_xy += x * y
            sum_x2 += x * x

        denom = m * sum_x2 - sum_x * sum_x
        if denom == 0:
            return 0.0

        slope = (m * sum_xy - sum_x * sum_y) / denom
        return slope

    def average_decision_time(self) -> float:
        """Average decision time in milliseconds, or 0.0 if no data."""
        if not self._decision_times:
            return 0.0
        return sum(self._decision_times) / len(self._decision_times)

    def get_session_info(self, current_hour: int | None = None) -> SessionInfo:
        """Return a snapshot of session timing information."""
        hour = current_hour if current_hour is not None else 12
        return SessionInfo(
            start_time=self.session_start,
            hands_played=self.hands_played,
            current_hour=hour,
        )

    def reset(self) -> None:
        """Reset the model for a new session."""
        self.session_start = time.time()
        self.hands_played = 0
        self._decision_times.clear()

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _duration_to_level(minutes: float) -> FatigueLevel:
        if minutes < 30:
            return FatigueLevel.FRESH
        if minutes < 90:
            return FatigueLevel.NORMAL
        if minutes < 180:
            return FatigueLevel.TIRED
        return FatigueLevel.EXHAUSTED

    @staticmethod
    def _bump_level(level: FatigueLevel) -> FatigueLevel:
        """Increase fatigue by one tier (for night penalty)."""
        order = [
            FatigueLevel.FRESH,
            FatigueLevel.NORMAL,
            FatigueLevel.TIRED,
            FatigueLevel.EXHAUSTED,
        ]
        idx = order.index(level)
        return order[min(idx + 1, len(order) - 1)]
