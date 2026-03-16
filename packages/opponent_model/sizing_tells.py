"""Bet Sizing Tell Detection.

Scientific basis:
- Ganzfried/Sandholm 2013: off-grid sizing reveals information
- Behavioral Economics anchoring: players anchor to previous sizing

Patterns:
- Small sizing (25-33% pot) with strong hand = trap/induce
- Large sizing (75-150% pot) as bluff = overcompensation
- Round numbers (2x, 3x) = inexperienced player
- Irregular sizes (37%, 67%) = player thinking about pot odds
- Same sizing always = leak; varied = balanced
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Sequence

from packages.common.types import ActionType

# ---------------------------------------------------------------------------
# Round-number bet fractions (pot-relative) that casual players gravitate to.
# ---------------------------------------------------------------------------
_ROUND_FRACTIONS: tuple[float, ...] = (
    0.25, 0.33, 0.50, 0.67, 0.75, 1.00, 1.50, 2.00,
)
_ROUND_TOLERANCE: float = 0.03

# Sizing buckets used for polarization analysis.
_SMALL_UPPER: float = 0.40   # <= 40% pot is "small"
_MEDIUM_LOWER: float = 0.40
_MEDIUM_UPPER: float = 0.80
_LARGE_LOWER: float = 0.80   # >= 80% pot is "large"


# ── Data classes ──────────────────────────────────────────────────────────

@dataclass
class SizingEvent:
    """A bet/raise with sizing information."""

    action: ActionType = ActionType.BET
    street: str = "flop"
    bet_fraction: float = 0.5  # as fraction of pot
    pot_size: float = 100.0
    bet_amount: float = 50.0


@dataclass
class SizingProfile:
    """Statistical profile of bet sizing patterns."""

    mean_fraction: float = 0.5
    std_fraction: float = 0.2
    count: int = 0
    uses_round_numbers: float = 0.0   # fraction of bets that are round numbers
    sizing_entropy: float = 0.0       # higher = more varied sizing


# ── Helpers ───────────────────────────────────────────────────────────────

def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _std(values: Sequence[float], mean: float) -> float:
    if len(values) < 2:
        return 0.0
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(variance)


def _entropy_of_histogram(counts: dict[int, int], total: int) -> float:
    """Shannon entropy (bits) of a histogram of bucket counts."""
    if total == 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        if c > 0:
            p = c / total
            ent -= p * math.log2(p)
    return ent


def _bucket_index(fraction: float, n_buckets: int = 10) -> int:
    """Map a bet fraction [0, +inf) into *n_buckets* discrete bins.

    Bin 0 represents [0, 0.2), bin 1 = [0.2, 0.4), ... bin 9 = [1.8, +inf).
    """
    idx = int(fraction / 0.2)
    return min(idx, n_buckets - 1)


# ── Main detector ─────────────────────────────────────────────────────────

class SizingTellDetector:
    """Detects patterns in bet sizing that reveal hand strength."""

    def __init__(self, window_size: int = 50) -> None:
        self._window_size = window_size
        # Per-street histories (most recent *window_size* events each).
        self._by_street: dict[str, deque[SizingEvent]] = {}
        # Flat history across all streets.
        self._all_events: deque[SizingEvent] = deque(maxlen=window_size)

        # Running correlation tracker (sizing vs ordinal index as a proxy
        # for hand-strength information – actual hand strength is unknown
        # so we use bet deviation from player mean as the signal).
        self._sum_x: float = 0.0   # cumulative bet_fraction
        self._sum_xx: float = 0.0  # cumulative bet_fraction^2
        self._n_corr: int = 0

    # ── Recording ─────────────────────────────────────────────────────

    def record_sizing(self, event: SizingEvent) -> None:
        """Record a sizing event."""
        street = event.street
        if street not in self._by_street:
            self._by_street[street] = deque(maxlen=self._window_size)
        self._by_street[street].append(event)
        self._all_events.append(event)

        # Update running stats for correlation.
        self._sum_x += event.bet_fraction
        self._sum_xx += event.bet_fraction ** 2
        self._n_corr += 1

    # ── Queries ───────────────────────────────────────────────────────

    def is_round_number(self, bet_fraction: float) -> bool:
        """Check if *bet_fraction* is close to a common round sizing."""
        for rnd in _ROUND_FRACTIONS:
            if abs(bet_fraction - rnd) <= _ROUND_TOLERANCE:
                return True
        return False

    def sizing_pattern(self, street: str | None = None) -> SizingProfile:
        """Get sizing profile, optionally filtered by street."""
        if street is not None:
            events = list(self._by_street.get(street, []))
        else:
            events = list(self._all_events)

        if not events:
            return SizingProfile()

        fractions = [e.bet_fraction for e in events]
        count = len(fractions)
        mean = _mean(fractions)
        std = _std(fractions, mean)

        # Round-number usage rate.
        round_count = sum(1 for f in fractions if self.is_round_number(f))
        round_rate = round_count / count

        # Entropy over coarse histogram buckets (10 bins).
        n_buckets = 10
        bucket_counts: dict[int, int] = {}
        for f in fractions:
            b = _bucket_index(f, n_buckets)
            bucket_counts[b] = bucket_counts.get(b, 0) + 1
        entropy = _entropy_of_histogram(bucket_counts, count)

        return SizingProfile(
            mean_fraction=mean,
            std_fraction=std,
            count=count,
            uses_round_numbers=round_rate,
            sizing_entropy=entropy,
        )

    # ── Inference ─────────────────────────────────────────────────────

    def infer_strength_from_sizing(self, event: SizingEvent) -> float:
        """Infer hand strength from sizing relative to player's baseline.

        Returns a value in [-1.0, 1.0].  Positive means likely strong.

        Heuristic rules
        ---------------
        1. Compute deviation = (bet_fraction - mean) / max(std, 0.05).
        2. Small deviation (|d| < 1) -> near zero (within normal range).
        3. Negative deviation (smaller than usual):
           - Might be a trap / slow-play -> lean positive (+0.3 scaled).
        4. Large positive deviation (bigger than usual):
           - Could be a bluff (overcompensation) -> lean negative (-0.3).
           - But *extremely* rare oversizes (|d| > 3) can be polarised
             toward the nuts -> lean positive (+0.5).
        5. Round-number adjustment: round sizing suggests less thought,
           nudge toward "less balanced" (slight positive since value-heavy
           players tend to pick round numbers).
        """
        profile = self.sizing_pattern(event.street)
        if profile.count < 3:
            return 0.0  # not enough data

        std = max(profile.std_fraction, 0.05)
        deviation = (event.bet_fraction - profile.mean_fraction) / std

        # Base signal from deviation.
        if abs(deviation) < 1.0:
            signal = 0.0
        elif deviation < -1.0:
            # Unusually small -> possible trap.
            signal = min(0.3 * abs(deviation) / 2.0, 0.6)
        elif deviation > 3.0:
            # Extremely large -> possible nuts overbet.
            signal = min(0.5 * (deviation - 3.0) / 2.0 + 0.2, 0.8)
        else:
            # Moderately large -> possible bluff.
            signal = -min(0.3 * deviation / 2.0, 0.5)

        # Round-number nudge (±0.05).
        if self.is_round_number(event.bet_fraction):
            if profile.uses_round_numbers < 0.3:
                # Player rarely uses round numbers but did now -> deliberate.
                signal += 0.05
            elif profile.uses_round_numbers > 0.7:
                # Player almost always rounds -> less info.
                pass

        return max(-1.0, min(1.0, signal))

    def polarization_score(self) -> float:
        """How polarized are the player's sizings?  0 = merged, 1 = very polarized.

        Polarized = uses very small AND very large, but not medium.
        Merged   = clusters around one size.
        """
        events = list(self._all_events)
        if len(events) < 5:
            return 0.0

        fractions = [e.bet_fraction for e in events]
        n_small = sum(1 for f in fractions if f <= _SMALL_UPPER)
        n_medium = sum(1 for f in fractions if _MEDIUM_LOWER < f <= _MEDIUM_UPPER)
        n_large = sum(1 for f in fractions if f > _LARGE_LOWER)
        total = len(fractions)

        small_rate = n_small / total
        medium_rate = n_medium / total
        large_rate = n_large / total

        # Polarization is high when small + large are high and medium is low.
        extremes = small_rate + large_rate
        if extremes == 0.0:
            return 0.0

        # Score = proportion of extremes minus medium, in [0, 1].
        raw = extremes * (1.0 - medium_rate)
        # Also require both extremes to be present (not just all-small).
        balance = 2.0 * min(small_rate, large_rate) / max(extremes, 1e-9)
        score = raw * balance
        return max(0.0, min(1.0, score))

    def exploitability_score(self) -> float:
        """How exploitable is the sizing pattern?  0 = balanced, 1 = very exploitable.

        Low entropy (same size always) = highly exploitable.
        High round-number usage = inexperienced = exploitable.
        """
        profile = self.sizing_pattern()
        if profile.count < 5:
            return 0.0

        # Max possible entropy for 10 bins is log2(10) ~ 3.32.
        max_entropy = math.log2(10)
        entropy_score = 1.0 - min(profile.sizing_entropy / max_entropy, 1.0)

        # Round-number score.
        round_score = profile.uses_round_numbers

        # Combine (entropy is more important).
        combined = 0.6 * entropy_score + 0.4 * round_score
        return max(0.0, min(1.0, combined))

    def get_correlation(self) -> float:
        """Correlation between bet size and inferred hand strength.

        Because actual hand strength is unknown at detection time, we use
        deviation from the player's mean sizing as a *proxy*.  The idea:
        if a player always bets bigger with strong hands, the deviation
        signal from ``infer_strength_from_sizing`` will track with
        bet_fraction.

        Returns a value in [-1, 1].  Positive = bigger bets with stronger
        hands (exploitable tell).  Near zero = balanced.
        """
        events = list(self._all_events)
        n = len(events)
        if n < 5:
            return 0.0

        xs: list[float] = []
        ys: list[float] = []
        for ev in events:
            xs.append(ev.bet_fraction)
            ys.append(self.infer_strength_from_sizing(ev))

        mean_x = _mean(xs)
        mean_y = _mean(ys)

        cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / n
        std_x = _std(xs, mean_x)
        std_y = _std(ys, mean_y)

        denom = std_x * std_y
        if denom < 1e-12:
            return 0.0

        r = cov / denom
        return max(-1.0, min(1.0, r))

    # ── Housekeeping ──────────────────────────────────────────────────

    def reset(self) -> None:
        """Discard all recorded events and reset internal state."""
        self._by_street.clear()
        self._all_events.clear()
        self._sum_x = 0.0
        self._sum_xx = 0.0
        self._n_corr = 0

    @property
    def events_recorded(self) -> int:
        """Total number of sizing events recorded (across all streets)."""
        return len(self._all_events)
