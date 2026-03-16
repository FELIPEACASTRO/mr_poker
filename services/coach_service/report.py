"""Coach Report Generator — Automatic session analysis with leak detection.

Generates structured coaching reports analyzing:
1. Session overview (hands, P&L, win rate)
2. Positional analysis (IP vs OOP performance)
3. Street-by-street analysis (preflop, flop, turn, river)
4. Leak detection (systematic errors)
5. Comparison to GTO baseline
6. Specific hand reviews (biggest winners/losers)
7. Behavioral patterns (tilt episodes, fatigue effects)
8. Skill progression (trend over sessions)
9. Actionable recommendations

Reference: docs/106_Analise_Estrategica_Avancada.md Section 1.4.2
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class LeakReport:
    """A detected systematic leak in the user's play."""

    leak_name: str
    severity: str  # "minor", "moderate", "major", "critical"
    description: str
    frequency: float  # how often the leak occurs (0-1)
    ev_cost_bb_per_hand: float  # estimated EV cost
    fix_suggestion: str
    example_hands: list[str]  # hand_ids demonstrating the leak


@dataclass
class StreetAnalysis:
    """Analysis of play on a single street."""

    street: str
    hands_seen: int
    aggression_frequency: float
    check_raise_frequency: float
    cbet_frequency: float
    fold_to_bet_frequency: float
    average_bet_size: float
    ev_per_hand: float
    gto_deviation_score: float  # 0-1, 0=perfect GTO


@dataclass
class PositionalAnalysis:
    """Analysis of play by position (IP or OOP)."""

    position: str  # "IP" or "OOP"
    hands_played: int
    vpip: float
    pfr: float
    win_rate_bb_per_hand: float
    aggression_factor: float


@dataclass
class CoachReport:
    """Complete coaching report for a session."""

    session_id: str
    generated_at: str

    # Overview
    total_hands: int
    session_duration_minutes: float
    net_profit_bb: float
    bb_per_hand: float  # win rate
    showdown_win_rate: float

    # Positional
    ip_analysis: PositionalAnalysis
    oop_analysis: PositionalAnalysis

    # Street-by-street
    street_analyses: list[StreetAnalysis]

    # Leaks
    leaks: list[LeakReport]

    # Behavioral
    tilt_episodes: int
    max_tilt_level: str
    fatigue_impact: str

    # Skill
    skill_estimate: float
    skill_label: str
    skill_trend: str  # "improving", "stable", "declining"

    # Recommendations
    top_recommendations: list[str]

    # Key hands
    biggest_winners: list[dict]
    biggest_losers: list[dict]
    most_interesting: list[dict]


# ---------------------------------------------------------------------------
# GTO baseline ranges for leak detection
# ---------------------------------------------------------------------------

_GTO_BASELINES = {
    "vpip_low": 0.22,
    "vpip_high": 0.32,
    "pfr_vpip_ratio_min": 0.60,
    "cbet_min": 0.50,
    "fold_to_cbet_max": 0.65,
    "overbet_max": 0.15,
    "limp_max": 0.10,
    "check_raise_min": 0.05,
    "showdown_win_rate_min": 0.45,
    "aggression_factor_min": 1.5,
}

# Severity mapping based on EV cost per hand (in BB)
_SEVERITY_THRESHOLDS = {
    "minor": 0.02,
    "moderate": 0.05,
    "major": 0.10,
    "critical": 0.20,
}


def _severity_from_ev_cost(ev_cost: float) -> str:
    """Classify severity based on estimated EV cost per hand."""
    if ev_cost >= _SEVERITY_THRESHOLDS["critical"]:
        return "critical"
    if ev_cost >= _SEVERITY_THRESHOLDS["major"]:
        return "major"
    if ev_cost >= _SEVERITY_THRESHOLDS["moderate"]:
        return "moderate"
    return "minor"


# ---------------------------------------------------------------------------
# Helper: safe extraction from hand dicts
# ---------------------------------------------------------------------------


def _get_actions(hand: dict) -> list[dict]:
    """Extract actions list from a hand dict, defaulting to empty."""
    return hand.get("actions", [])


def _get_profit(hand: dict) -> float:
    """Extract profit in BB from a hand dict."""
    return float(hand.get("profit_bb", hand.get("result_bb", 0.0)))


def _get_position(hand: dict) -> str:
    """Return 'IP' or 'OOP' for the hand."""
    return hand.get("position", "unknown")


def _get_hand_id(hand: dict) -> str:
    """Extract hand ID."""
    return hand.get("hand_id", hand.get("id", "unknown"))


def _actions_on_street(actions: list[dict], street: str) -> list[dict]:
    """Filter actions to a specific street."""
    return [a for a in actions if a.get("street", "") == street]


def _is_aggressive(action: dict) -> bool:
    """Whether an action is aggressive (bet, raise, all_in)."""
    return action.get("action", action.get("action_type", "")) in (
        "bet", "raise", "all_in",
    )


def _is_passive(action: dict) -> bool:
    """Whether an action is passive (call, check)."""
    return action.get("action", action.get("action_type", "")) in (
        "call", "check",
    )


def _is_fold(action: dict) -> bool:
    """Whether an action is a fold."""
    return action.get("action", action.get("action_type", "")) == "fold"


def _bet_fraction(action: dict) -> float:
    """Extract bet fraction (size relative to pot)."""
    return float(action.get("bet_fraction", action.get("size_pot", 0.0)))


# ---------------------------------------------------------------------------
# CoachReportGenerator
# ---------------------------------------------------------------------------


class CoachReportGenerator:
    """Generates coaching reports from session data.

    Analyzes decision traces, hand results, and behavioral signals
    to produce actionable coaching feedback.

    Usage::

        generator = CoachReportGenerator()
        report = generator.generate(session_data)
        print(generator.format_text(report))
    """

    def __init__(self) -> None:
        pass

    def generate(self, session_data: dict) -> CoachReport:
        """Generate a complete coaching report from session data.

        Parameters
        ----------
        session_data:
            Dictionary containing:
            - ``hands``: list of hand dicts with actions, results, traces
            - ``session_id``: str
            - ``duration_minutes``: float
            - ``behavioral_signals``: optional list of behavioral snapshots

        Returns
        -------
        A fully populated :class:`CoachReport`.
        """
        hands = session_data.get("hands", [])
        session_id = session_data.get("session_id", "unknown")
        duration = float(session_data.get("duration_minutes", 0.0))
        behavioral = session_data.get("behavioral_signals", [])

        # Core analyses
        overview = self._analyze_overview(hands)
        ip_analysis = self._analyze_position(hands, "IP")
        oop_analysis = self._analyze_position(hands, "OOP")
        street_analyses = self._analyze_streets(hands)
        leaks = self._detect_leaks(hands, street_analyses)
        key_hands = self._find_key_hands(hands)
        recommendations = self._generate_recommendations(leaks, street_analyses)

        # Behavioral analysis
        tilt_episodes, max_tilt_level = self._analyze_tilt(behavioral)
        fatigue_impact = self._analyze_fatigue(behavioral)

        # Skill estimation
        skill_estimate, skill_label = self._estimate_skill(
            overview, leaks, street_analyses
        )
        skill_trend = self._estimate_trend(session_data)

        total_hands = overview["total_hands"]

        return CoachReport(
            session_id=session_id,
            generated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            total_hands=total_hands,
            session_duration_minutes=duration,
            net_profit_bb=overview["net_profit_bb"],
            bb_per_hand=overview["bb_per_hand"],
            showdown_win_rate=overview["showdown_win_rate"],
            ip_analysis=ip_analysis,
            oop_analysis=oop_analysis,
            street_analyses=street_analyses,
            leaks=leaks,
            tilt_episodes=tilt_episodes,
            max_tilt_level=max_tilt_level,
            fatigue_impact=fatigue_impact,
            skill_estimate=skill_estimate,
            skill_label=skill_label,
            skill_trend=skill_trend,
            top_recommendations=recommendations,
            biggest_winners=key_hands["biggest_winners"],
            biggest_losers=key_hands["biggest_losers"],
            most_interesting=key_hands["most_interesting"],
        )

    # ------------------------------------------------------------------
    # Analysis methods
    # ------------------------------------------------------------------

    def _analyze_overview(self, hands: list[dict]) -> dict:
        """Compute session overview statistics.

        Returns
        -------
        Dict with keys: total_hands, net_profit_bb, bb_per_hand,
        showdown_win_rate.
        """
        total_hands = len(hands)
        if total_hands == 0:
            return {
                "total_hands": 0,
                "net_profit_bb": 0.0,
                "bb_per_hand": 0.0,
                "showdown_win_rate": 0.0,
            }

        net_profit = sum(_get_profit(h) for h in hands)
        bb_per_hand = net_profit / total_hands

        # Showdown stats
        showdown_hands = [
            h for h in hands if h.get("went_to_showdown", False)
        ]
        if showdown_hands:
            showdown_wins = sum(
                1 for h in showdown_hands if _get_profit(h) > 0.0
            )
            showdown_win_rate = showdown_wins / len(showdown_hands)
        else:
            showdown_win_rate = 0.0

        return {
            "total_hands": total_hands,
            "net_profit_bb": net_profit,
            "bb_per_hand": bb_per_hand,
            "showdown_win_rate": showdown_win_rate,
        }

    def _analyze_position(
        self, hands: list[dict], position: str
    ) -> PositionalAnalysis:
        """Analyze performance for a specific position (IP or OOP).

        Parameters
        ----------
        hands:
            All hand dicts for the session.
        position:
            Either ``"IP"`` or ``"OOP"``.

        Returns
        -------
        A :class:`PositionalAnalysis` for the given position.
        """
        pos_hands = [h for h in hands if _get_position(h) == position]
        n = len(pos_hands)

        if n == 0:
            return PositionalAnalysis(
                position=position,
                hands_played=0,
                vpip=0.0,
                pfr=0.0,
                win_rate_bb_per_hand=0.0,
                aggression_factor=0.0,
            )

        # VPIP: voluntarily put money in pot preflop
        vpip_count = 0
        pfr_count = 0
        aggressive_count = 0
        passive_count = 0

        for h in pos_hands:
            actions = _get_actions(h)
            preflop = _actions_on_street(actions, "preflop")

            # Check if player voluntarily entered the pot
            for a in preflop:
                act = a.get("action", a.get("action_type", ""))
                if act in ("call", "bet", "raise", "all_in"):
                    vpip_count += 1
                    break

            # PFR: raised or bet preflop
            for a in preflop:
                act = a.get("action", a.get("action_type", ""))
                if act in ("raise", "bet", "all_in"):
                    pfr_count += 1
                    break

            # Overall aggression
            for a in actions:
                if _is_aggressive(a):
                    aggressive_count += 1
                elif a.get("action", a.get("action_type", "")) == "call":
                    passive_count += 1

        vpip = vpip_count / n
        pfr = pfr_count / n
        total_profit = sum(_get_profit(h) for h in pos_hands)
        win_rate = total_profit / n

        if passive_count > 0:
            aggression_factor = aggressive_count / passive_count
        else:
            aggression_factor = float(aggressive_count) if aggressive_count > 0 else 0.0

        return PositionalAnalysis(
            position=position,
            hands_played=n,
            vpip=round(vpip, 3),
            pfr=round(pfr, 3),
            win_rate_bb_per_hand=round(win_rate, 3),
            aggression_factor=round(aggression_factor, 2),
        )

    def _analyze_streets(self, hands: list[dict]) -> list[StreetAnalysis]:
        """Street-by-street analysis of play.

        Returns a list of :class:`StreetAnalysis` for preflop, flop,
        turn, and river.
        """
        streets = ["preflop", "flop", "turn", "river"]
        results: list[StreetAnalysis] = []

        for street in streets:
            hands_with_street = []
            agg_actions = 0
            passive_actions = 0
            fold_actions = 0
            total_actions = 0
            check_raise_count = 0
            cbet_opportunities = 0
            cbet_made = 0
            fold_to_bet_opportunities = 0
            fold_to_bet_count = 0
            bet_sizes: list[float] = []

            for h in hands:
                actions = _actions_on_street(_get_actions(h), street)
                if not actions:
                    continue
                hands_with_street.append(h)

                had_check = False
                for a in actions:
                    total_actions += 1
                    if _is_aggressive(a):
                        agg_actions += 1
                        frac = _bet_fraction(a)
                        if frac > 0:
                            bet_sizes.append(frac)
                        # Check-raise: check followed by raise
                        if had_check:
                            check_raise_count += 1
                    elif _is_fold(a):
                        fold_actions += 1
                    elif _is_passive(a):
                        passive_actions += 1
                        if a.get("action", a.get("action_type", "")) == "check":
                            had_check = True

                # C-bet detection (postflop only): was aggressor preflop
                # and acted first on this street
                if street != "preflop" and h.get("was_preflop_aggressor", False):
                    cbet_opportunities += 1
                    street_acts = actions
                    if street_acts and _is_aggressive(street_acts[0]):
                        cbet_made += 1

                # Fold to bet: faced a bet and folded
                if any(_is_aggressive(a) for a in actions):
                    fold_to_bet_opportunities += 1
                    if any(_is_fold(a) for a in actions):
                        fold_to_bet_count += 1

            n_hands = len(hands_with_street)
            if n_hands == 0:
                results.append(StreetAnalysis(
                    street=street,
                    hands_seen=0,
                    aggression_frequency=0.0,
                    check_raise_frequency=0.0,
                    cbet_frequency=0.0,
                    fold_to_bet_frequency=0.0,
                    average_bet_size=0.0,
                    ev_per_hand=0.0,
                    gto_deviation_score=0.0,
                ))
                continue

            agg_freq = agg_actions / max(1, total_actions)
            cr_freq = check_raise_count / max(1, n_hands)
            cbet_freq = cbet_made / max(1, cbet_opportunities) if cbet_opportunities > 0 else 0.0
            fold_freq = fold_to_bet_count / max(1, fold_to_bet_opportunities) if fold_to_bet_opportunities > 0 else 0.0
            avg_bet = sum(bet_sizes) / len(bet_sizes) if bet_sizes else 0.0

            # EV per hand on this street: sum of profits for hands that
            # reached this street / number of such hands
            ev_per_hand = sum(_get_profit(h) for h in hands_with_street) / n_hands

            # GTO deviation: composite of how far key stats are from baseline
            deviations: list[float] = []
            if street == "preflop":
                # VPIP deviation handled at position level; use aggression here
                deviations.append(abs(agg_freq - 0.35) / 0.35)
            else:
                if cbet_opportunities > 0:
                    deviations.append(
                        abs(cbet_freq - 0.55) / 0.55
                    )
                deviations.append(abs(fold_freq - 0.45) / 0.45)
                deviations.append(abs(agg_freq - 0.40) / 0.40)

            gto_dev = min(1.0, sum(deviations) / max(1, len(deviations)))

            results.append(StreetAnalysis(
                street=street,
                hands_seen=n_hands,
                aggression_frequency=round(agg_freq, 3),
                check_raise_frequency=round(cr_freq, 3),
                cbet_frequency=round(cbet_freq, 3),
                fold_to_bet_frequency=round(fold_freq, 3),
                average_bet_size=round(avg_bet, 3),
                ev_per_hand=round(ev_per_hand, 3),
                gto_deviation_score=round(gto_dev, 3),
            ))

        return results

    def _detect_leaks(
        self,
        hands: list[dict],
        street_analyses: list[StreetAnalysis],
    ) -> list[LeakReport]:
        """Identify systematic leaks in the user's play.

        Implements 10 leak detection rules comparing session statistics
        against GTO baselines.
        """
        leaks: list[LeakReport] = []
        total = len(hands)
        if total == 0:
            return leaks

        # Pre-compute session-wide stats
        vpip_count = 0
        pfr_count = 0
        limp_count = 0
        total_aggressive = 0
        total_calls = 0
        showdown_hands = []
        tilt_loss_hands: list[dict] = []

        for h in hands:
            actions = _get_actions(h)
            preflop = _actions_on_street(actions, "preflop")

            entered_pot = False
            raised_preflop = False
            limped = False

            for a in preflop:
                act = a.get("action", a.get("action_type", ""))
                if act in ("call", "bet", "raise", "all_in"):
                    entered_pot = True
                if act in ("raise", "bet", "all_in"):
                    raised_preflop = True
                # Limp: calling preflop without a prior raise in hand actions
                if act == "call" and not raised_preflop:
                    limped = True

            if entered_pot:
                vpip_count += 1
            if raised_preflop:
                pfr_count += 1
            if limped:
                limp_count += 1

            for a in actions:
                if _is_aggressive(a):
                    total_aggressive += 1
                elif a.get("action", a.get("action_type", "")) == "call":
                    total_calls += 1

            if h.get("went_to_showdown", False):
                showdown_hands.append(h)

            # Tilt-tagged hands
            if h.get("tilt_active", False):
                tilt_loss_hands.append(h)

        vpip = vpip_count / total
        pfr = pfr_count / total
        limp_rate = limp_count / total

        # Build a lookup for street analyses
        street_map = {sa.street: sa for sa in street_analyses}

        # Helper to find example hands for a leak
        def _example_ids(filter_fn: Any, max_n: int = 3) -> list[str]:
            examples = [_get_hand_id(h) for h in hands if filter_fn(h)]
            return examples[:max_n]

        # --- Leak 1: Fold to C-bet too often (>65%) ---
        for street_name in ("flop", "turn", "river"):
            sa = street_map.get(street_name)
            if sa and sa.hands_seen >= 10 and sa.fold_to_bet_frequency > _GTO_BASELINES["fold_to_cbet_max"]:
                freq = sa.fold_to_bet_frequency
                ev_cost = (freq - _GTO_BASELINES["fold_to_cbet_max"]) * 0.5  # estimated
                leaks.append(LeakReport(
                    leak_name=f"Fold to C-bet too often ({street_name})",
                    severity=_severity_from_ev_cost(ev_cost),
                    description=(
                        f"Folding to bets {freq:.0%} of the time on the {street_name} "
                        f"(baseline max: {_GTO_BASELINES['fold_to_cbet_max']:.0%}). "
                        f"Opponents can exploit this by bluffing more."
                    ),
                    frequency=freq,
                    ev_cost_bb_per_hand=round(ev_cost, 3),
                    fix_suggestion=(
                        f"Develop a defending range on the {street_name}. Add more "
                        f"calls with medium-strength hands and check-raise bluffs."
                    ),
                    example_hands=_example_ids(
                        lambda h, s=street_name: any(
                            _is_fold(a) for a in _actions_on_street(_get_actions(h), s)
                        )
                    ),
                ))

        # --- Leak 2: Not C-betting enough (<50%) ---
        for street_name in ("flop", "turn"):
            sa = street_map.get(street_name)
            if sa and sa.hands_seen >= 10 and sa.cbet_frequency < _GTO_BASELINES["cbet_min"] and sa.cbet_frequency > 0:
                freq = sa.cbet_frequency
                ev_cost = (_GTO_BASELINES["cbet_min"] - freq) * 0.4
                leaks.append(LeakReport(
                    leak_name=f"Not C-betting enough ({street_name})",
                    severity=_severity_from_ev_cost(ev_cost),
                    description=(
                        f"C-betting only {freq:.0%} on the {street_name} "
                        f"(baseline min: {_GTO_BASELINES['cbet_min']:.0%}). "
                        f"Missing value and bluffing opportunities."
                    ),
                    frequency=freq,
                    ev_cost_bb_per_hand=round(ev_cost, 3),
                    fix_suggestion=(
                        f"As the preflop aggressor, c-bet more often on the "
                        f"{street_name} with both value hands and balanced bluffs."
                    ),
                    example_hands=_example_ids(
                        lambda h: h.get("was_preflop_aggressor", False)
                    ),
                ))

        # --- Leak 3: Overbet frequency too high (>15%) ---
        overbet_count = 0
        total_bets = 0
        for h in hands:
            for a in _get_actions(h):
                if _is_aggressive(a):
                    total_bets += 1
                    if _bet_fraction(a) > 1.0:
                        overbet_count += 1

        if total_bets >= 20:
            overbet_freq = overbet_count / total_bets
            if overbet_freq > _GTO_BASELINES["overbet_max"]:
                ev_cost = (overbet_freq - _GTO_BASELINES["overbet_max"]) * 0.8
                leaks.append(LeakReport(
                    leak_name="Overbet frequency too high",
                    severity=_severity_from_ev_cost(ev_cost),
                    description=(
                        f"Overbetting {overbet_freq:.0%} of bets "
                        f"(baseline max: {_GTO_BASELINES['overbet_max']:.0%}). "
                        f"Opponents with strong calling ranges will punish this."
                    ),
                    frequency=overbet_freq,
                    ev_cost_bb_per_hand=round(ev_cost, 3),
                    fix_suggestion=(
                        "Use overbets selectively with polarized ranges "
                        "(strong value or pure bluffs), not with medium hands."
                    ),
                    example_hands=_example_ids(
                        lambda h: any(
                            _bet_fraction(a) > 1.0 for a in _get_actions(h)
                            if _is_aggressive(a)
                        )
                    ),
                ))

        # --- Leak 4: Limp frequency too high (>10%) ---
        if total >= 20 and limp_rate > _GTO_BASELINES["limp_max"]:
            ev_cost = (limp_rate - _GTO_BASELINES["limp_max"]) * 0.6
            leaks.append(LeakReport(
                leak_name="Limp frequency too high",
                severity=_severity_from_ev_cost(ev_cost),
                description=(
                    f"Limping {limp_rate:.0%} of hands "
                    f"(baseline max: {_GTO_BASELINES['limp_max']:.0%}). "
                    f"Open-limping is exploitable and gives up initiative."
                ),
                frequency=limp_rate,
                ev_cost_bb_per_hand=round(ev_cost, 3),
                fix_suggestion=(
                    "Replace limps with raises. If a hand is not worth "
                    "raising, it is usually not worth playing."
                ),
                example_hands=_example_ids(
                    lambda h: any(
                        a.get("action", a.get("action_type", "")) == "call"
                        for a in _actions_on_street(_get_actions(h), "preflop")
                    )
                ),
            ))

        # --- Leak 5: Check-raise frequency too low (<5%) ---
        for street_name in ("flop", "turn", "river"):
            sa = street_map.get(street_name)
            if sa and sa.hands_seen >= 20 and sa.check_raise_frequency < _GTO_BASELINES["check_raise_min"]:
                freq = sa.check_raise_frequency
                ev_cost = (_GTO_BASELINES["check_raise_min"] - freq) * 0.3
                leaks.append(LeakReport(
                    leak_name=f"Check-raise frequency too low ({street_name})",
                    severity=_severity_from_ev_cost(ev_cost),
                    description=(
                        f"Check-raising only {freq:.0%} on the {street_name} "
                        f"(baseline min: {_GTO_BASELINES['check_raise_min']:.0%}). "
                        f"A low check-raise frequency makes your checking range weak."
                    ),
                    frequency=freq,
                    ev_cost_bb_per_hand=round(ev_cost, 3),
                    fix_suggestion=(
                        f"Add check-raises on the {street_name} with strong hands "
                        f"and balanced bluffs to protect your checking range."
                    ),
                    example_hands=[],
                ))

        # --- Leak 6: Showdown win rate too low (<45%) ---
        if len(showdown_hands) >= 10:
            sd_wins = sum(1 for h in showdown_hands if _get_profit(h) > 0)
            sd_rate = sd_wins / len(showdown_hands)
            if sd_rate < _GTO_BASELINES["showdown_win_rate_min"]:
                ev_cost = (_GTO_BASELINES["showdown_win_rate_min"] - sd_rate) * 1.0
                leaks.append(LeakReport(
                    leak_name="Showdown win rate too low",
                    severity=_severity_from_ev_cost(ev_cost),
                    description=(
                        f"Winning only {sd_rate:.0%} at showdown "
                        f"(baseline min: {_GTO_BASELINES['showdown_win_rate_min']:.0%}). "
                        f"This suggests calling too wide or not value-betting enough."
                    ),
                    frequency=sd_rate,
                    ev_cost_bb_per_hand=round(ev_cost, 3),
                    fix_suggestion=(
                        "Tighten your calling ranges, especially on the river. "
                        "Fold more marginal hands when facing large bets."
                    ),
                    example_hands=_example_ids(
                        lambda h: h.get("went_to_showdown", False)
                        and _get_profit(h) < 0.0
                    ),
                ))

        # --- Leak 7: VPIP too high or too low ---
        if total >= 20:
            if vpip > _GTO_BASELINES["vpip_high"]:
                excess = vpip - _GTO_BASELINES["vpip_high"]
                ev_cost = excess * 0.8
                leaks.append(LeakReport(
                    leak_name="VPIP too high",
                    severity=_severity_from_ev_cost(ev_cost),
                    description=(
                        f"VPIP is {vpip:.0%} (baseline max: "
                        f"{_GTO_BASELINES['vpip_high']:.0%}). "
                        f"Playing too many hands preflop."
                    ),
                    frequency=vpip,
                    ev_cost_bb_per_hand=round(ev_cost, 3),
                    fix_suggestion=(
                        "Tighten your preflop range. Fold more marginal "
                        "hands, especially from early positions."
                    ),
                    example_hands=_example_ids(
                        lambda h: _get_profit(h) < -2.0
                    ),
                ))
            elif vpip < _GTO_BASELINES["vpip_low"]:
                deficit = _GTO_BASELINES["vpip_low"] - vpip
                ev_cost = deficit * 0.5
                leaks.append(LeakReport(
                    leak_name="VPIP too low",
                    severity=_severity_from_ev_cost(ev_cost),
                    description=(
                        f"VPIP is {vpip:.0%} (baseline min: "
                        f"{_GTO_BASELINES['vpip_low']:.0%}). "
                        f"Playing too tight, missing profitable spots."
                    ),
                    frequency=vpip,
                    ev_cost_bb_per_hand=round(ev_cost, 3),
                    fix_suggestion=(
                        "Widen your preflop range from late positions. "
                        "Add suited connectors, suited aces, and Broadway hands."
                    ),
                    example_hands=[],
                ))

        # --- Leak 8: PFR too low relative to VPIP ---
        if total >= 20 and vpip > 0.05:
            pfr_ratio = pfr / vpip
            if pfr_ratio < _GTO_BASELINES["pfr_vpip_ratio_min"]:
                ev_cost = (_GTO_BASELINES["pfr_vpip_ratio_min"] - pfr_ratio) * 0.5
                leaks.append(LeakReport(
                    leak_name="PFR too low relative to VPIP",
                    severity=_severity_from_ev_cost(ev_cost),
                    description=(
                        f"PFR/VPIP ratio is {pfr_ratio:.2f} "
                        f"(baseline min: {_GTO_BASELINES['pfr_vpip_ratio_min']:.2f}). "
                        f"Too much flat-calling preflop instead of raising."
                    ),
                    frequency=pfr_ratio,
                    ev_cost_bb_per_hand=round(ev_cost, 3),
                    fix_suggestion=(
                        "When entering a pot, prefer raising over calling. "
                        "Raising takes initiative and narrows opponent ranges."
                    ),
                    example_hands=_example_ids(
                        lambda h: any(
                            a.get("action", a.get("action_type", "")) == "call"
                            for a in _actions_on_street(_get_actions(h), "preflop")
                        )
                    ),
                ))

        # --- Leak 9: Post-flop aggression too low (AF < 1.5) ---
        if total_calls > 0:
            af = total_aggressive / total_calls
        else:
            af = float(total_aggressive)

        if total >= 20 and total_calls >= 10 and af < _GTO_BASELINES["aggression_factor_min"]:
            ev_cost = (_GTO_BASELINES["aggression_factor_min"] - af) * 0.15
            leaks.append(LeakReport(
                leak_name="Post-flop aggression too low",
                severity=_severity_from_ev_cost(ev_cost),
                description=(
                    f"Aggression factor is {af:.2f} "
                    f"(baseline min: {_GTO_BASELINES['aggression_factor_min']:.2f}). "
                    f"Playing too passively postflop."
                ),
                frequency=af / 5.0,  # normalize to 0-1 range
                ev_cost_bb_per_hand=round(ev_cost, 3),
                fix_suggestion=(
                    "Bet and raise more postflop, especially with strong "
                    "hands and draws. Passive play gives opponents free cards."
                ),
                example_hands=_example_ids(
                    lambda h: any(
                        _is_passive(a) and a.get("street", "") != "preflop"
                        for a in _get_actions(h)
                    )
                ),
            ))

        # --- Leak 10: Tilt-induced losses ---
        if tilt_loss_hands:
            tilt_losses = sum(
                _get_profit(h) for h in tilt_loss_hands if _get_profit(h) < 0
            )
            tilt_freq = len(tilt_loss_hands) / total
            ev_cost = abs(tilt_losses) / total if total > 0 else 0.0
            if ev_cost > 0.01:
                leaks.append(LeakReport(
                    leak_name="Tilt-induced losses",
                    severity=_severity_from_ev_cost(ev_cost),
                    description=(
                        f"Lost {abs(tilt_losses):.1f} BB during tilt episodes "
                        f"across {len(tilt_loss_hands)} hands. "
                        f"Emotional play is the most costly leak."
                    ),
                    frequency=tilt_freq,
                    ev_cost_bb_per_hand=round(ev_cost, 3),
                    fix_suggestion=(
                        "Implement a stop-loss rule and take mandatory breaks "
                        "after bad beats. Consider the responsible gaming module."
                    ),
                    example_hands=_example_ids(
                        lambda h: h.get("tilt_active", False) and _get_profit(h) < 0
                    ),
                ))

        # Sort leaks by EV cost (most costly first)
        leaks.sort(key=lambda lk: lk.ev_cost_bb_per_hand, reverse=True)
        return leaks

    def _find_key_hands(self, hands: list[dict]) -> dict:
        """Find biggest winners, losers, and most interesting hands.

        Returns
        -------
        Dict with ``biggest_winners``, ``biggest_losers``, and
        ``most_interesting`` lists (up to 5 each).
        """
        if not hands:
            return {
                "biggest_winners": [],
                "biggest_losers": [],
                "most_interesting": [],
            }

        sorted_by_profit = sorted(hands, key=_get_profit)

        # Biggest losers (most negative first)
        biggest_losers = [
            {
                "hand_id": _get_hand_id(h),
                "profit_bb": _get_profit(h),
                "went_to_showdown": h.get("went_to_showdown", False),
            }
            for h in sorted_by_profit[:5]
            if _get_profit(h) < 0
        ]

        # Biggest winners (most positive first)
        biggest_winners = [
            {
                "hand_id": _get_hand_id(h),
                "profit_bb": _get_profit(h),
                "went_to_showdown": h.get("went_to_showdown", False),
            }
            for h in reversed(sorted_by_profit[-5:])
            if _get_profit(h) > 0
        ]

        # Most interesting: hands with the most actions (complex pots)
        hands_by_actions = sorted(
            hands,
            key=lambda h: len(_get_actions(h)),
            reverse=True,
        )
        most_interesting = [
            {
                "hand_id": _get_hand_id(h),
                "profit_bb": _get_profit(h),
                "action_count": len(_get_actions(h)),
                "went_to_showdown": h.get("went_to_showdown", False),
            }
            for h in hands_by_actions[:5]
            if len(_get_actions(h)) > 3
        ]

        return {
            "biggest_winners": biggest_winners,
            "biggest_losers": biggest_losers,
            "most_interesting": most_interesting,
        }

    def _generate_recommendations(
        self,
        leaks: list[LeakReport],
        street_analyses: list[StreetAnalysis],
    ) -> list[str]:
        """Generate prioritized recommendations from detected leaks.

        Produces up to 5 actionable recommendations, ordered by
        expected EV impact.
        """
        recommendations: list[str] = []

        # Primary: address the top leaks by EV cost
        for leak in leaks[:3]:
            recommendations.append(
                f"[{leak.severity.upper()}] {leak.leak_name}: {leak.fix_suggestion}"
            )

        # Street-specific recommendations
        for sa in street_analyses:
            if sa.hands_seen >= 10 and sa.gto_deviation_score > 0.5:
                recommendations.append(
                    f"Review your {sa.street} strategy — GTO deviation "
                    f"score is {sa.gto_deviation_score:.2f}. Study common "
                    f"{sa.street} spots with a solver."
                )

        # If few leaks, add general improvement advice
        if len(recommendations) < 3:
            recommendations.append(
                "Session looks solid. Focus on refining bet sizing and "
                "range construction for marginal spots."
            )
            recommendations.append(
                "Review the 3-5 biggest losing hands to identify any "
                "patterns in your decision-making."
            )

        return recommendations[:5]

    # ------------------------------------------------------------------
    # Behavioral analysis
    # ------------------------------------------------------------------

    def _analyze_tilt(self, behavioral: list[dict]) -> tuple[int, str]:
        """Count tilt episodes and find max tilt level from behavioral signals.

        Returns
        -------
        Tuple of (episode_count, max_tilt_level_string).
        """
        if not behavioral:
            return 0, "normal"

        tilt_levels = {"normal": 0, "mild_tilt": 1, "full_tilt": 2, "steaming": 3}
        reverse_levels = {v: k for k, v in tilt_levels.items()}

        episodes = 0
        max_level = 0
        was_tilting = False

        for signal in behavioral:
            level_str = signal.get("tilt_state", signal.get("tilt_level", "normal"))
            level = tilt_levels.get(level_str, 0)
            max_level = max(max_level, level)

            if level >= 1 and not was_tilting:
                episodes += 1
                was_tilting = True
            elif level == 0:
                was_tilting = False

        return episodes, reverse_levels.get(max_level, "normal")

    def _analyze_fatigue(self, behavioral: list[dict]) -> str:
        """Analyze fatigue impact from behavioral signals.

        Returns a descriptive string: 'none', 'mild', 'moderate', 'severe'.
        """
        if not behavioral:
            return "none"

        fatigue_scores = [
            float(s.get("fatigue_score", s.get("fatigue", 0.0)))
            for s in behavioral
            if "fatigue_score" in s or "fatigue" in s
        ]

        if not fatigue_scores:
            return "none"

        max_fatigue = max(fatigue_scores)
        avg_fatigue = sum(fatigue_scores) / len(fatigue_scores)

        if max_fatigue >= 0.7 or avg_fatigue >= 0.5:
            return "severe"
        if max_fatigue >= 0.5 or avg_fatigue >= 0.3:
            return "moderate"
        if max_fatigue >= 0.3:
            return "mild"
        return "none"

    # ------------------------------------------------------------------
    # Skill estimation
    # ------------------------------------------------------------------

    def _estimate_skill(
        self,
        overview: dict,
        leaks: list[LeakReport],
        street_analyses: list[StreetAnalysis],
    ) -> tuple[float, str]:
        """Estimate player skill level 0-100.

        Combines win rate, leak severity, and GTO deviation into a
        composite skill estimate.

        Returns
        -------
        Tuple of (skill_score, skill_label).
        """
        # Base from win rate: 50 + (bb/hand * 200), clamped to [20, 80]
        bb_per_hand = overview.get("bb_per_hand", 0.0)
        base = 50.0 + bb_per_hand * 200.0
        base = max(20.0, min(80.0, base))

        # Leak penalty: each leak reduces score based on severity
        severity_penalties = {"minor": 1.0, "moderate": 3.0, "major": 5.0, "critical": 8.0}
        leak_penalty = sum(
            severity_penalties.get(lk.severity, 2.0) for lk in leaks
        )
        leak_penalty = min(30.0, leak_penalty)

        # GTO deviation bonus/penalty
        avg_gto_dev = 0.0
        if street_analyses:
            devs = [sa.gto_deviation_score for sa in street_analyses if sa.hands_seen > 0]
            if devs:
                avg_gto_dev = sum(devs) / len(devs)
        gto_penalty = avg_gto_dev * 15.0  # max ~15 point penalty

        score = max(0.0, min(100.0, base - leak_penalty - gto_penalty))

        # Classify
        if score >= 80:
            label = "advanced"
        elif score >= 60:
            label = "intermediate"
        elif score >= 40:
            label = "developing"
        else:
            label = "beginner"

        return round(score, 1), label

    def _estimate_trend(self, session_data: dict) -> str:
        """Estimate skill trend from historical data.

        Uses ``previous_sessions`` in session_data if available.

        Returns ``"improving"``, ``"stable"``, or ``"declining"``.
        """
        previous = session_data.get("previous_sessions", [])
        if len(previous) < 3:
            return "stable"

        # Compare average win rate of last 3 sessions vs earlier ones
        recent_wr = [
            float(s.get("bb_per_hand", 0.0)) for s in previous[-3:]
        ]
        earlier_wr = [
            float(s.get("bb_per_hand", 0.0)) for s in previous[:-3]
        ]

        if not earlier_wr:
            return "stable"

        avg_recent = sum(recent_wr) / len(recent_wr)
        avg_earlier = sum(earlier_wr) / len(earlier_wr)

        diff = avg_recent - avg_earlier
        if diff > 0.02:
            return "improving"
        if diff < -0.02:
            return "declining"
        return "stable"

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def format_text(self, report: CoachReport) -> str:
        """Format report as human-readable text.

        Produces a structured multi-section text report suitable for
        display in a terminal or text file.
        """
        lines: list[str] = []
        sep = "=" * 60

        lines.append(sep)
        lines.append("COACHING REPORT")
        lines.append(sep)
        lines.append(f"Session: {report.session_id}")
        lines.append(f"Generated: {report.generated_at}")
        lines.append("")

        # Overview
        lines.append("--- SESSION OVERVIEW ---")
        lines.append(f"Hands played:     {report.total_hands}")
        lines.append(f"Duration:         {report.session_duration_minutes:.0f} min")
        lines.append(f"Net profit:       {report.net_profit_bb:+.1f} BB")
        lines.append(f"Win rate:         {report.bb_per_hand:+.3f} BB/hand")
        lines.append(f"Showdown win %:   {report.showdown_win_rate:.0%}")
        lines.append("")

        # Positional
        lines.append("--- POSITIONAL ANALYSIS ---")
        for pa in (report.ip_analysis, report.oop_analysis):
            if pa.hands_played > 0:
                lines.append(
                    f"  {pa.position}: {pa.hands_played} hands, "
                    f"VPIP={pa.vpip:.0%}, PFR={pa.pfr:.0%}, "
                    f"WR={pa.win_rate_bb_per_hand:+.3f} BB/h, "
                    f"AF={pa.aggression_factor:.2f}"
                )
        lines.append("")

        # Streets
        lines.append("--- STREET ANALYSIS ---")
        for sa in report.street_analyses:
            if sa.hands_seen > 0:
                lines.append(
                    f"  {sa.street.upper()}: {sa.hands_seen} hands, "
                    f"Agg={sa.aggression_frequency:.0%}, "
                    f"CR={sa.check_raise_frequency:.0%}, "
                    f"Cbet={sa.cbet_frequency:.0%}, "
                    f"F2B={sa.fold_to_bet_frequency:.0%}, "
                    f"EV={sa.ev_per_hand:+.3f}, "
                    f"GTO-dev={sa.gto_deviation_score:.2f}"
                )
        lines.append("")

        # Leaks
        lines.append("--- LEAKS DETECTED ---")
        if report.leaks:
            for lk in report.leaks:
                lines.append(
                    f"  [{lk.severity.upper()}] {lk.leak_name} "
                    f"(cost: {lk.ev_cost_bb_per_hand:+.3f} BB/hand)"
                )
                lines.append(f"    {lk.description}")
                lines.append(f"    Fix: {lk.fix_suggestion}")
        else:
            lines.append("  No significant leaks detected.")
        lines.append("")

        # Behavioral
        lines.append("--- BEHAVIORAL ---")
        lines.append(f"Tilt episodes:  {report.tilt_episodes}")
        lines.append(f"Max tilt level: {report.max_tilt_level}")
        lines.append(f"Fatigue impact: {report.fatigue_impact}")
        lines.append("")

        # Skill
        lines.append("--- SKILL ESTIMATE ---")
        lines.append(f"Score: {report.skill_estimate:.1f}/100 ({report.skill_label})")
        lines.append(f"Trend: {report.skill_trend}")
        lines.append("")

        # Recommendations
        lines.append("--- RECOMMENDATIONS ---")
        for i, rec in enumerate(report.top_recommendations, 1):
            lines.append(f"  {i}. {rec}")
        lines.append("")

        # Key hands
        lines.append("--- KEY HANDS ---")
        if report.biggest_winners:
            lines.append("  Biggest Winners:")
            for h in report.biggest_winners:
                lines.append(f"    {h['hand_id']}: {h['profit_bb']:+.1f} BB")
        if report.biggest_losers:
            lines.append("  Biggest Losers:")
            for h in report.biggest_losers:
                lines.append(f"    {h['hand_id']}: {h['profit_bb']:+.1f} BB")
        if report.most_interesting:
            lines.append("  Most Complex:")
            for h in report.most_interesting:
                lines.append(
                    f"    {h['hand_id']}: {h['profit_bb']:+.1f} BB "
                    f"({h['action_count']} actions)"
                )
        lines.append("")
        lines.append(sep)

        return "\n".join(lines)

    def format_dict(self, report: CoachReport) -> dict:
        """Format report as a serializable dictionary.

        Converts all dataclass fields to plain dicts/lists suitable
        for JSON serialization.
        """
        return {
            "session_id": report.session_id,
            "generated_at": report.generated_at,
            "overview": {
                "total_hands": report.total_hands,
                "session_duration_minutes": report.session_duration_minutes,
                "net_profit_bb": report.net_profit_bb,
                "bb_per_hand": report.bb_per_hand,
                "showdown_win_rate": report.showdown_win_rate,
            },
            "positional": {
                "ip": {
                    "position": report.ip_analysis.position,
                    "hands_played": report.ip_analysis.hands_played,
                    "vpip": report.ip_analysis.vpip,
                    "pfr": report.ip_analysis.pfr,
                    "win_rate_bb_per_hand": report.ip_analysis.win_rate_bb_per_hand,
                    "aggression_factor": report.ip_analysis.aggression_factor,
                },
                "oop": {
                    "position": report.oop_analysis.position,
                    "hands_played": report.oop_analysis.hands_played,
                    "vpip": report.oop_analysis.vpip,
                    "pfr": report.oop_analysis.pfr,
                    "win_rate_bb_per_hand": report.oop_analysis.win_rate_bb_per_hand,
                    "aggression_factor": report.oop_analysis.aggression_factor,
                },
            },
            "street_analyses": [
                {
                    "street": sa.street,
                    "hands_seen": sa.hands_seen,
                    "aggression_frequency": sa.aggression_frequency,
                    "check_raise_frequency": sa.check_raise_frequency,
                    "cbet_frequency": sa.cbet_frequency,
                    "fold_to_bet_frequency": sa.fold_to_bet_frequency,
                    "average_bet_size": sa.average_bet_size,
                    "ev_per_hand": sa.ev_per_hand,
                    "gto_deviation_score": sa.gto_deviation_score,
                }
                for sa in report.street_analyses
            ],
            "leaks": [
                {
                    "leak_name": lk.leak_name,
                    "severity": lk.severity,
                    "description": lk.description,
                    "frequency": lk.frequency,
                    "ev_cost_bb_per_hand": lk.ev_cost_bb_per_hand,
                    "fix_suggestion": lk.fix_suggestion,
                    "example_hands": lk.example_hands,
                }
                for lk in report.leaks
            ],
            "behavioral": {
                "tilt_episodes": report.tilt_episodes,
                "max_tilt_level": report.max_tilt_level,
                "fatigue_impact": report.fatigue_impact,
            },
            "skill": {
                "estimate": report.skill_estimate,
                "label": report.skill_label,
                "trend": report.skill_trend,
            },
            "recommendations": report.top_recommendations,
            "key_hands": {
                "biggest_winners": report.biggest_winners,
                "biggest_losers": report.biggest_losers,
                "most_interesting": report.most_interesting,
            },
        }
