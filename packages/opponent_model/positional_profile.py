"""Positional Profiling — Per-position opponent statistics.

Scientific basis: Pluribus exploitation patterns, Patrick/Spiderdime.
Players vary dramatically by position. A "nit in EP but LAG on BTN" is
a skilled player; "same VPIP everywhere" is a fish.

Tracks VPIP, PFR, 3-bet, c-bet, fold-to-cbet per position.
Computes positional awareness score to distinguish skilled regulars
from recreational players.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from packages.common.types import ActionType


POSITIONS = ["EP", "MP", "CO", "BTN", "SB", "BB"]

# Positions considered "in position" postflop (late position / button)
_IP_POSITIONS = {"CO", "BTN"}
# Positions considered "out of position" postflop
_OOP_POSITIONS = {"EP", "MP", "SB", "BB"}

# Rough GTO VPIP benchmarks by position (6-max)
_OPTIMAL_VPIP: dict[str, float] = {
    "EP": 0.15,
    "MP": 0.18,
    "CO": 0.27,
    "BTN": 0.40,
    "SB": 0.30,
    "BB": 0.35,
}


@dataclass
class PositionalStats:
    """Stats for a single position."""

    hands: int = 0
    vpip_count: int = 0
    pfr_count: int = 0
    three_bet_count: int = 0
    three_bet_opps: int = 0
    cbet_count: int = 0
    cbet_opps: int = 0
    fold_to_cbet_count: int = 0
    fold_to_cbet_opps: int = 0
    total_aggressive: int = 0
    total_passive: int = 0
    total_folds: int = 0

    # -- derived properties ------------------------------------------------

    @property
    def vpip(self) -> float:
        """Voluntarily put money in pot percentage."""
        return self.vpip_count / max(1, self.hands)

    @property
    def pfr(self) -> float:
        """Pre-flop raise percentage."""
        return self.pfr_count / max(1, self.hands)

    @property
    def three_bet_pct(self) -> float:
        return self.three_bet_count / max(1, self.three_bet_opps)

    @property
    def cbet_pct(self) -> float:
        return self.cbet_count / max(1, self.cbet_opps)

    @property
    def fold_to_cbet_pct(self) -> float:
        return self.fold_to_cbet_count / max(1, self.fold_to_cbet_opps)

    @property
    def aggression_factor(self) -> float:
        return self.total_aggressive / max(1, self.total_passive)

    @property
    def fold_pct(self) -> float:
        total = self.total_aggressive + self.total_passive + self.total_folds
        return self.total_folds / max(1, total)


class PositionalProfiler:
    """Tracks per-position statistics for an opponent.

    Usage::

        profiler = PositionalProfiler()
        profiler.record_hand("BTN")
        profiler.record_action("BTN", ActionType.RAISE, street="preflop")
        profiler.record_action("BTN", ActionType.BET, street="flop", is_cbet=True)
        stats = profiler.get_stats("BTN")
    """

    def __init__(self) -> None:
        self.positions: dict[str, PositionalStats] = {
            p: PositionalStats() for p in POSITIONS
        }

    # -- recording ---------------------------------------------------------

    def record_hand(self, position: str) -> None:
        """Record that a new hand was played in *position*."""
        stats = self._resolve(position)
        stats.hands += 1

    def record_action(
        self,
        position: str,
        action: ActionType,
        *,
        street: str = "preflop",
        is_three_bet_opp: bool = False,
        is_cbet: bool = False,
        facing_cbet: bool = False,
    ) -> None:
        """Record a single action in *position*.

        Parameters
        ----------
        position:
            One of POSITIONS (case-insensitive lookup supported).
        action:
            The ActionType performed.
        street:
            "preflop", "flop", "turn", or "river".
        is_three_bet_opp:
            True if this was an opportunity to 3-bet.
        is_cbet:
            True if this action is a continuation bet opportunity
            (player was preflop aggressor on the flop).
        facing_cbet:
            True if opponent is facing a c-bet.
        """
        stats = self._resolve(position)

        # Aggression tracking
        if action in {ActionType.BET, ActionType.RAISE, ActionType.ALL_IN}:
            stats.total_aggressive += 1
        elif action in {ActionType.CHECK, ActionType.CALL}:
            stats.total_passive += 1
        elif action == ActionType.FOLD:
            stats.total_folds += 1

        # Preflop specifics
        if street == "preflop":
            if action in {
                ActionType.CALL,
                ActionType.BET,
                ActionType.RAISE,
                ActionType.ALL_IN,
            }:
                stats.vpip_count += 1
            if action in {ActionType.RAISE, ActionType.BET, ActionType.ALL_IN}:
                stats.pfr_count += 1

            # 3-bet tracking
            if is_three_bet_opp:
                stats.three_bet_opps += 1
                if action in {ActionType.RAISE, ActionType.ALL_IN}:
                    stats.three_bet_count += 1

        # C-bet tracking
        if is_cbet:
            stats.cbet_opps += 1
            if action in {ActionType.BET, ActionType.RAISE, ActionType.ALL_IN}:
                stats.cbet_count += 1

        if facing_cbet:
            stats.fold_to_cbet_opps += 1
            if action == ActionType.FOLD:
                stats.fold_to_cbet_count += 1

    # -- queries -----------------------------------------------------------

    def get_stats(self, position: str) -> PositionalStats:
        """Return stats for *position*."""
        return self._resolve(position)

    def positional_awareness_score(self) -> float:
        """Return 0..1 score of how much VPIP varies across positions.

        0 = plays the same everywhere (fish tendency).
        1 = high positional variance (skilled regular).

        Requires at least 5 hands in every position; returns 0.0 otherwise.
        """
        vpips: list[float] = []
        for pos in POSITIONS:
            s = self.positions[pos]
            if s.hands < 5:
                return 0.0
            vpips.append(s.vpip)

        if not vpips:
            return 0.0

        mean = sum(vpips) / len(vpips)
        variance = sum((v - mean) ** 2 for v in vpips) / len(vpips)
        # Normalise: a stdev of ~0.12 is "very aware", cap at 1.0
        raw = math.sqrt(variance) / 0.12
        return min(1.0, max(0.0, raw))

    def weakest_position(self) -> str:
        """Position where opponent deviates most from optimal (highest excess VPIP).

        Returns the position string, e.g. ``"EP"``.
        Falls back to ``"EP"`` when no data is available.
        """
        worst_pos = "EP"
        worst_delta = -1.0
        for pos in POSITIONS:
            s = self.positions[pos]
            if s.hands < 5:
                continue
            optimal = _OPTIMAL_VPIP.get(pos, 0.25)
            delta = s.vpip - optimal  # positive = too loose
            if delta > worst_delta:
                worst_delta = delta
                worst_pos = pos
        return worst_pos

    def ip_vs_oop_ratio(self) -> float:
        """Ratio of aggression factor in-position vs out-of-position.

        >1 = more aggressive IP (standard, skilled).
        <1 = more aggressive OOP (backwards, exploitable).
        Returns 1.0 when insufficient data.
        """
        ip_agg, ip_pass = 0, 0
        for pos in _IP_POSITIONS:
            s = self.positions[pos]
            ip_agg += s.total_aggressive
            ip_pass += s.total_passive

        oop_agg, oop_pass = 0, 0
        for pos in _OOP_POSITIONS:
            s = self.positions[pos]
            oop_agg += s.total_aggressive
            oop_pass += s.total_passive

        ip_af = ip_agg / max(1, ip_pass)
        oop_af = oop_agg / max(1, oop_pass)

        if oop_af == 0.0:
            return 1.0
        return ip_af / oop_af

    def deviation_from_optimal(self, position: str) -> float:
        """Return signed VPIP deviation from GTO benchmark for *position*.

        Positive = too loose, negative = too tight.
        """
        s = self._resolve(position)
        if s.hands < 5:
            return 0.0
        return s.vpip - _OPTIMAL_VPIP.get(position.upper(), 0.25)

    def total_hands(self) -> int:
        """Sum of hands across all positions."""
        return sum(s.hands for s in self.positions.values())

    def reset(self) -> None:
        """Clear all tracked data."""
        self.positions = {p: PositionalStats() for p in POSITIONS}

    # -- internals ---------------------------------------------------------

    def _resolve(self, position: str) -> PositionalStats:
        key = position.upper()
        if key not in self.positions:
            self.positions[key] = PositionalStats()
        return self.positions[key]
