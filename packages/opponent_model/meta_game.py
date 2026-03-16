"""Meta-Game Dynamics Tracking.

Detects if an opponent is adjusting to our play style over time.

Uses a sliding-window approach: compare a *recent* window of hands
against a longer *baseline* window.  Statistically significant shifts
indicate the opponent is adapting.

Thinking-level estimation:
- Level 0: plays own cards, ignores opponents.
- Level 1: reacts to opponent tendencies (adapts once).
- Level 2: anticipates that we are adapting to them (counter-adapts).

Scientific basis:
- Bowling et al. (2015) on convergence in imperfect-information games.
- Bard et al. (2013) on opponent modelling in repeated games.
"""

from __future__ import annotations

import enum
import math
from collections import deque
from dataclasses import dataclass, field


class AdaptationState(enum.Enum):
    """Current meta-game state of the opponent."""

    STATIC = "static"
    ADAPTING = "adapting"
    COUNTER_ADAPTING = "counter_adapting"


@dataclass
class WindowStats:
    """Aggregated stats from a window of hands."""

    vpip: float = 0.0
    pfr: float = 0.0
    aggression: float = 0.0
    fold_to_cbet: float = 0.0
    hands: int = 0


# Threshold (Euclidean distance in stat-space) to declare "adapting"
_ADAPT_THRESHOLD = 0.10
# Threshold to declare "counter-adapting" (reversal)
_COUNTER_THRESHOLD = 0.08


class MetaGameTracker:
    """Detects whether an opponent is adjusting to our play.

    Call :meth:`record_stats_snapshot` once per hand with the opponent's
    running stats.  Then query :meth:`detect_adaptation` or
    :meth:`recommended_response` to decide how to adjust.

    Usage::

        meta = MetaGameTracker()
        for hand in session:
            meta.record_stats_snapshot(
                vpip=opp.vpip, pfr=opp.pfr,
                aggression=opp.aggression_factor,
                fold_to_cbet=opp.fold_to_cbet_pct,
            )
        state = meta.detect_adaptation()
    """

    def __init__(
        self,
        recent_window: int = 20,
        baseline_window: int = 100,
    ) -> None:
        self._recent_size = recent_window
        self._baseline_size = baseline_window
        self._snapshots: deque[tuple[float, float, float, float]] = deque(
            maxlen=baseline_window
        )
        # Track direction of previous shift for counter-adaptation detection
        self._prev_delta: tuple[float, float, float, float] | None = None

    # -- recording ---------------------------------------------------------

    def record_stats_snapshot(
        self,
        vpip: float,
        pfr: float,
        aggression: float,
        fold_to_cbet: float,
    ) -> None:
        """Record a stats snapshot (call once per hand)."""
        self._snapshots.append((vpip, pfr, aggression, fold_to_cbet))

    # -- queries -----------------------------------------------------------

    def detect_adaptation(self) -> AdaptationState:
        """Compare recent vs baseline stats.

        Returns:
            STATIC if no significant shift.
            ADAPTING if recent stats differ significantly from baseline.
            COUNTER_ADAPTING if the shift has reversed direction.
        """
        baseline = self._compute_window(0, self._baseline_size)
        recent = self._compute_window_recent()

        if baseline.hands < self._baseline_size // 2 or recent.hands < self._recent_size // 2:
            return AdaptationState.STATIC

        delta = self._stat_delta(baseline, recent)
        dist = self._euclidean(delta)

        if dist < _ADAPT_THRESHOLD:
            self._prev_delta = None
            return AdaptationState.STATIC

        # Check for counter-adaptation: shift reversed from previous
        if self._prev_delta is not None:
            dot = sum(a * b for a, b in zip(delta, self._prev_delta))
            if dot < -_COUNTER_THRESHOLD:
                self._prev_delta = delta
                return AdaptationState.COUNTER_ADAPTING

        self._prev_delta = delta
        return AdaptationState.ADAPTING

    def adaptation_magnitude(self) -> float:
        """How much has opponent changed? 0 = static, 1 = completely different.

        Normalised Euclidean distance between recent and baseline windows.
        """
        baseline = self._compute_window(0, self._baseline_size)
        recent = self._compute_window_recent()

        if baseline.hands < 5 or recent.hands < 5:
            return 0.0

        delta = self._stat_delta(baseline, recent)
        dist = self._euclidean(delta)
        # Normalise: a distance of 0.4 in stat-space ≈ "completely different"
        return min(1.0, dist / 0.40)

    def thinking_level(self) -> int:
        """Estimate opponent's level of thinking.

        0 = plays own cards (no adaptation detected).
        1 = considers opponent (adapts once).
        2 = considers what opponent thinks (counter-adapts).
        """
        state = self.detect_adaptation()
        if state == AdaptationState.COUNTER_ADAPTING:
            return 2
        if state == AdaptationState.ADAPTING:
            return 1
        return 0

    def recommended_response(self) -> str:
        """Suggest a strategic response.

        STATIC -> exploit aggressively.
        ADAPTING -> reduce exploitation, move toward GTO.
        COUNTER_ADAPTING -> play GTO to avoid levelling wars.
        """
        state = self.detect_adaptation()
        responses = {
            AdaptationState.STATIC: "exploit",
            AdaptationState.ADAPTING: "reduce_exploit",
            AdaptationState.COUNTER_ADAPTING: "gto",
        }
        return responses[state]

    def reset(self) -> None:
        """Clear all tracked data."""
        self._snapshots.clear()
        self._prev_delta = None

    @property
    def snapshots_recorded(self) -> int:
        return len(self._snapshots)

    # -- internals ---------------------------------------------------------

    def _compute_window(self, start: int, size: int) -> WindowStats:
        """Compute average stats over a slice of the snapshot buffer."""
        items = list(self._snapshots)
        end = min(start + size, len(items))
        window = items[start:end]
        if not window:
            return WindowStats()

        n = len(window)
        return WindowStats(
            vpip=sum(s[0] for s in window) / n,
            pfr=sum(s[1] for s in window) / n,
            aggression=sum(s[2] for s in window) / n,
            fold_to_cbet=sum(s[3] for s in window) / n,
            hands=n,
        )

    def _compute_window_recent(self) -> WindowStats:
        """Compute stats over the most recent window."""
        items = list(self._snapshots)
        start = max(0, len(items) - self._recent_size)
        window = items[start:]
        if not window:
            return WindowStats()

        n = len(window)
        return WindowStats(
            vpip=sum(s[0] for s in window) / n,
            pfr=sum(s[1] for s in window) / n,
            aggression=sum(s[2] for s in window) / n,
            fold_to_cbet=sum(s[3] for s in window) / n,
            hands=n,
        )

    @staticmethod
    def _stat_delta(
        baseline: WindowStats, recent: WindowStats
    ) -> tuple[float, float, float, float]:
        return (
            recent.vpip - baseline.vpip,
            recent.pfr - baseline.pfr,
            # Normalise aggression (typically 0-5) to 0-1 scale
            (recent.aggression - baseline.aggression) / 5.0,
            recent.fold_to_cbet - baseline.fold_to_cbet,
        )

    @staticmethod
    def _euclidean(delta: tuple[float, float, float, float]) -> float:
        return math.sqrt(sum(d * d for d in delta))
