"""SAD Profiler — Search and Destroy opponent leak detection.

Rapidly identifies the top exploitable leaks in an opponent's play
within <100 hands, then generates targeted counter-strategies.

Inspired by: Patrick bot (Spiderdime, 2025) — profitable in microstakes
real money games via rapid leak identification and exploitation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from packages.common.types import ActionType
from packages.opponent_model.classifier import PlayerStats

logger = logging.getLogger(__name__)


class LeakType(str, Enum):
    """Types of exploitable leaks."""
    FOLDS_TOO_MUCH = "folds_too_much"
    CALLS_TOO_MUCH = "calls_too_much"
    NEVER_BLUFFS = "never_bluffs"
    OVER_BLUFFS = "over_bluffs"
    NO_CBET = "no_cbet"
    ALWAYS_CBETS = "always_cbets"
    FOLD_TO_CBET = "fold_to_cbet"
    NEVER_3BETS = "never_3bets"
    OVER_3BETS = "over_3bets"
    FOLD_TO_3BET = "fold_to_3bet"
    PASSIVE_POSTFLOP = "passive_postflop"
    OVERBET_HAPPY = "overbet_happy"
    FIT_OR_FOLD = "fit_or_fold"
    SHOWDOWN_BOUND = "showdown_bound"
    WEAK_TURN = "weak_turn"
    WEAK_RIVER = "weak_river"


@dataclass
class Leak:
    """A detected leak with severity and counter-strategy."""
    leak_type: LeakType
    severity: float  # 0.0 (minor) to 1.0 (massive)
    description: str
    counter_strategy: str
    # Evidence stats
    observed_value: float
    expected_gto_range: tuple[float, float]  # (min, max) GTO range
    sample_size: int

    @property
    def is_significant(self) -> bool:
        """Whether we have enough data to be confident in this leak."""
        return self.sample_size >= 15 and self.severity >= 0.3


@dataclass
class SADProfile:
    """Complete Search-and-Destroy profile for an opponent."""
    player_id: str
    leaks: list[Leak] = field(default_factory=list)
    hands_analyzed: int = 0
    overall_exploit_score: float = 0.0  # 0-1, higher = more exploitable

    @property
    def top_leaks(self) -> list[Leak]:
        """Top 3 most severe and significant leaks."""
        significant = [l for l in self.leaks if l.is_significant]
        return sorted(significant, key=lambda l: l.severity, reverse=True)[:3]

    @property
    def has_actionable_leaks(self) -> bool:
        """Whether we have enough data for meaningful exploitation."""
        return len(self.top_leaks) >= 1


# GTO baseline ranges for leak detection
GTO_RANGES = {
    "vpip": (0.22, 0.30),        # 22-30% is standard
    "pfr": (0.17, 0.25),          # 17-25%
    "3bet": (0.06, 0.12),         # 6-12%
    "fold_to_3bet": (0.40, 0.60), # 40-60%
    "cbet": (0.55, 0.75),         # 55-75%
    "fold_to_cbet": (0.35, 0.50), # 35-50%
    "aggression_factor": (2.0, 4.0),  # 2.0-4.0
    "wtsd": (0.22, 0.30),         # 22-30%
    "fold_pct": (0.30, 0.45),     # 30-45% overall
}


class SADProfiler:
    """Search and Destroy profiler for rapid leak identification.

    Analyzes an opponent's stats and identifies exploitable patterns
    in as few as 30-50 hands. Generates specific counter-strategies
    for each detected leak.
    """

    def __init__(self, min_hands: int = 20) -> None:
        self.min_hands = min_hands

    def analyze(self, player_id: str, stats: PlayerStats) -> SADProfile:
        """Analyze player stats and return a SAD profile with detected leaks."""
        profile = SADProfile(player_id=player_id, hands_analyzed=stats.total_hands)

        if stats.total_hands < self.min_hands:
            return profile

        # Run all leak detectors
        detectors = [
            self._check_folds_too_much,
            self._check_calls_too_much,
            self._check_cbet_leaks,
            self._check_3bet_leaks,
            self._check_passive_postflop,
            self._check_showdown_tendencies,
            self._check_street_weakness,
        ]

        for detector in detectors:
            leaks = detector(stats)
            profile.leaks.extend(leaks)

        # Compute overall exploit score
        if profile.leaks:
            significant_severities = [
                l.severity for l in profile.leaks if l.is_significant
            ]
            if significant_severities:
                profile.overall_exploit_score = min(
                    1.0,
                    sum(significant_severities) / len(significant_severities),
                )

        return profile

    def _severity(self, value: float, gto_range: tuple[float, float]) -> float:
        """Compute severity of a deviation from GTO range."""
        low, high = gto_range
        if low <= value <= high:
            return 0.0
        if value < low:
            deviation = (low - value) / max(low, 0.01)
        else:
            deviation = (value - high) / max(1.0 - high, 0.01)
        return min(1.0, deviation)

    def _check_folds_too_much(self, stats: PlayerStats) -> list[Leak]:
        leaks = []
        fold_pct = stats.fold_pct
        gto = GTO_RANGES["fold_pct"]
        if fold_pct > gto[1]:
            leaks.append(Leak(
                leak_type=LeakType.FOLDS_TOO_MUCH,
                severity=self._severity(fold_pct, gto),
                description=f"Folds {fold_pct:.0%} overall (GTO: {gto[0]:.0%}-{gto[1]:.0%})",
                counter_strategy="Increase bluff frequency. Bet more frequently on all streets. "
                    "Use larger sizing to maximize fold equity.",
                observed_value=fold_pct,
                expected_gto_range=gto,
                sample_size=stats.total_hands,
            ))
        return leaks

    def _check_calls_too_much(self, stats: PlayerStats) -> list[Leak]:
        leaks = []
        if stats.total_hands < self.min_hands:
            return leaks

        # VPIP too high = calling too much
        vpip = stats.vpip
        gto = GTO_RANGES["vpip"]
        if vpip > gto[1] + 0.10:  # 10% above GTO max
            leaks.append(Leak(
                leak_type=LeakType.CALLS_TOO_MUCH,
                severity=self._severity(vpip, gto),
                description=f"VPIP {vpip:.0%} — plays way too many hands (GTO: {gto[0]:.0%}-{gto[1]:.0%})",
                counter_strategy="Value bet thinner. Reduce bluff frequency. "
                    "They'll call with marginal hands, so bet for value more.",
                observed_value=vpip,
                expected_gto_range=gto,
                sample_size=stats.total_hands,
            ))
        return leaks

    def _check_cbet_leaks(self, stats: PlayerStats) -> list[Leak]:
        leaks = []
        if stats.cbet_opportunities < 5:
            return leaks

        cbet = stats.cbet_pct
        gto = GTO_RANGES["cbet"]

        if cbet < gto[0]:
            leaks.append(Leak(
                leak_type=LeakType.NO_CBET,
                severity=self._severity(cbet, gto),
                description=f"C-bets only {cbet:.0%} (GTO: {gto[0]:.0%}-{gto[1]:.0%})",
                counter_strategy="Float more. Probe bet when checked to. "
                    "Their check means weakness — attack it.",
                observed_value=cbet,
                expected_gto_range=gto,
                sample_size=stats.cbet_opportunities,
            ))
        elif cbet > gto[1] + 0.10:
            leaks.append(Leak(
                leak_type=LeakType.ALWAYS_CBETS,
                severity=self._severity(cbet, gto),
                description=f"C-bets {cbet:.0%} — over-cbetting (GTO: {gto[0]:.0%}-{gto[1]:.0%})",
                counter_strategy="Check-raise more flops. Float and raise turns. "
                    "They're cbetting air too often.",
                observed_value=cbet,
                expected_gto_range=gto,
                sample_size=stats.cbet_opportunities,
            ))

        # Fold to c-bet
        if stats.cbet_faced >= 5:
            ftcb = stats.fold_to_cbet_pct
            gto_ftcb = GTO_RANGES["fold_to_cbet"]
            if ftcb > gto_ftcb[1]:
                leaks.append(Leak(
                    leak_type=LeakType.FOLD_TO_CBET,
                    severity=self._severity(ftcb, gto_ftcb),
                    description=f"Folds to c-bet {ftcb:.0%} (GTO: {gto_ftcb[0]:.0%}-{gto_ftcb[1]:.0%})",
                    counter_strategy="C-bet more frequently vs this player. "
                        "Use smaller sizing since they fold too much anyway.",
                    observed_value=ftcb,
                    expected_gto_range=gto_ftcb,
                    sample_size=stats.cbet_faced,
                ))

        return leaks

    def _check_3bet_leaks(self, stats: PlayerStats) -> list[Leak]:
        leaks = []
        if stats.three_bet_opportunities < 5:
            return leaks

        three_bet = stats.three_bet_pct
        gto = GTO_RANGES["3bet"]

        if three_bet < gto[0] - 0.02:
            leaks.append(Leak(
                leak_type=LeakType.NEVER_3BETS,
                severity=self._severity(three_bet, gto),
                description=f"3-bets only {three_bet:.0%} (GTO: {gto[0]:.0%}-{gto[1]:.0%})",
                counter_strategy="Open wider. Steal more. Their flat-call range "
                    "is wide but weak — c-bet aggressively.",
                observed_value=three_bet,
                expected_gto_range=gto,
                sample_size=stats.three_bet_opportunities,
            ))
        elif three_bet > gto[1] + 0.05:
            leaks.append(Leak(
                leak_type=LeakType.OVER_3BETS,
                severity=self._severity(three_bet, gto),
                description=f"3-bets {three_bet:.0%} — too aggressive (GTO: {gto[0]:.0%}-{gto[1]:.0%})",
                counter_strategy="4-bet bluff more. Trap with premium hands. "
                    "Call 3-bets wider in position.",
                observed_value=three_bet,
                expected_gto_range=gto,
                sample_size=stats.three_bet_opportunities,
            ))

        # Fold to 3-bet
        ft3b = stats.fold_to_3bet_pct
        gto_ft3b = GTO_RANGES["fold_to_3bet"]
        if ft3b > gto_ft3b[1]:
            leaks.append(Leak(
                leak_type=LeakType.FOLD_TO_3BET,
                severity=self._severity(ft3b, gto_ft3b),
                description=f"Folds to 3-bet {ft3b:.0%} (GTO: {gto_ft3b[0]:.0%}-{gto_ft3b[1]:.0%})",
                counter_strategy="3-bet bluff more. Use smaller 3-bet sizing. "
                    "They're giving up too much preflop equity.",
                observed_value=ft3b,
                expected_gto_range=gto_ft3b,
                sample_size=stats.three_bet_opportunities,
            ))

        return leaks

    def _check_passive_postflop(self, stats: PlayerStats) -> list[Leak]:
        leaks = []
        af = stats.aggression_factor
        gto = GTO_RANGES["aggression_factor"]

        if af < gto[0] and stats.total_postflop_actions >= 10:
            leaks.append(Leak(
                leak_type=LeakType.PASSIVE_POSTFLOP,
                severity=self._severity(af, gto),
                description=f"Aggression factor {af:.1f} — too passive (GTO: {gto[0]:.1f}-{gto[1]:.1f})",
                counter_strategy="Bet for thin value more. Bluff less (they're calling). "
                    "Check-back more marginal hands (they don't raise).",
                observed_value=af,
                expected_gto_range=gto,
                sample_size=stats.total_postflop_actions,
            ))
        return leaks

    def _check_showdown_tendencies(self, stats: PlayerStats) -> list[Leak]:
        leaks = []
        if stats.total_hands < self.min_hands:
            return leaks

        wtsd = stats.wtsd
        gto = GTO_RANGES["wtsd"]

        if wtsd > gto[1] + 0.05:
            leaks.append(Leak(
                leak_type=LeakType.SHOWDOWN_BOUND,
                severity=self._severity(wtsd, gto),
                description=f"WTSD {wtsd:.0%} — goes to showdown too often (GTO: {gto[0]:.0%}-{gto[1]:.0%})",
                counter_strategy="Value bet big on river. Bluff less on river. "
                    "They'll call with marginal hands at showdown.",
                observed_value=wtsd,
                expected_gto_range=gto,
                sample_size=stats.total_hands,
            ))

        if wtsd < gto[0] - 0.05 and stats.total_hands >= 30:
            leaks.append(Leak(
                leak_type=LeakType.FIT_OR_FOLD,
                severity=self._severity(wtsd, gto),
                description=f"WTSD {wtsd:.0%} — fit-or-fold style (GTO: {gto[0]:.0%}-{gto[1]:.0%})",
                counter_strategy="Barrel more streets. Multi-street bluffs work well. "
                    "They give up on later streets without strong hands.",
                observed_value=wtsd,
                expected_gto_range=gto,
                sample_size=stats.total_hands,
            ))

        return leaks

    def _check_street_weakness(self, stats: PlayerStats) -> list[Leak]:
        leaks = []

        if stats.turn_actions >= 10:
            turn_agg = stats.turn_aggression / max(stats.turn_actions, 1)
            if turn_agg < 0.30:
                leaks.append(Leak(
                    leak_type=LeakType.WEAK_TURN,
                    severity=min(1.0, (0.30 - turn_agg) / 0.30),
                    description=f"Turn aggression {turn_agg:.0%} — gives up too easily on turn",
                    counter_strategy="Double barrel more. Probe bet turns when checked to. "
                        "Their turn check is very exploitable.",
                    observed_value=turn_agg,
                    expected_gto_range=(0.35, 0.55),
                    sample_size=stats.turn_actions,
                ))

        if stats.river_actions >= 10:
            river_agg = stats.river_aggression / max(stats.river_actions, 1)
            if river_agg < 0.25:
                leaks.append(Leak(
                    leak_type=LeakType.WEAK_RIVER,
                    severity=min(1.0, (0.25 - river_agg) / 0.25),
                    description=f"River aggression {river_agg:.0%} — rarely bets river",
                    counter_strategy="Bluff-catch more on river. They only bet with strong hands. "
                        "Probe bet rivers when checked to.",
                    observed_value=river_agg,
                    expected_gto_range=(0.30, 0.50),
                    sample_size=stats.river_actions,
                ))

        return leaks

    def generate_exploit_adjustments(
        self, profile: SADProfile
    ) -> dict[str, float]:
        """Generate numerical adjustments based on detected leaks.

        Returns a dict of adjustment factors that can be applied to
        a baseline strategy to exploit the opponent.

        Keys: bluff_frequency, value_bet_frequency, fold_frequency,
              cbet_frequency, 3bet_frequency, call_frequency
        """
        adjustments: dict[str, float] = {
            "bluff_frequency": 0.0,
            "value_bet_frequency": 0.0,
            "fold_frequency": 0.0,
            "cbet_frequency": 0.0,
            "three_bet_frequency": 0.0,
            "call_frequency": 0.0,
        }

        for leak in profile.top_leaks:
            s = leak.severity

            if leak.leak_type == LeakType.FOLDS_TOO_MUCH:
                adjustments["bluff_frequency"] += 0.3 * s
                adjustments["cbet_frequency"] += 0.2 * s

            elif leak.leak_type == LeakType.CALLS_TOO_MUCH:
                adjustments["value_bet_frequency"] += 0.3 * s
                adjustments["bluff_frequency"] -= 0.2 * s

            elif leak.leak_type == LeakType.FOLD_TO_CBET:
                adjustments["cbet_frequency"] += 0.3 * s

            elif leak.leak_type == LeakType.FOLD_TO_3BET:
                adjustments["three_bet_frequency"] += 0.25 * s

            elif leak.leak_type == LeakType.PASSIVE_POSTFLOP:
                adjustments["value_bet_frequency"] += 0.2 * s

            elif leak.leak_type == LeakType.SHOWDOWN_BOUND:
                adjustments["value_bet_frequency"] += 0.25 * s
                adjustments["bluff_frequency"] -= 0.15 * s

            elif leak.leak_type == LeakType.FIT_OR_FOLD:
                adjustments["bluff_frequency"] += 0.25 * s

        return adjustments
