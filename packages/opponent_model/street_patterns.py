"""Multi-Street Behavioral Pattern Tracking.

Tracks sequential betting patterns across streets to detect:
- Barrel frequency (single, double, triple)
- Check-raise tendencies per street
- Probe bets (betting when the aggressor checks)
- Floats (call flop, bet turn when checked to)
- Delayed c-bets (check flop as aggressor, bet turn)
- Give-up frequency

Scientific basis: multi-street play is where most EV is won or lost
in deep-stacked poker. Pluribus and Libratus both learned that
sequential bet patterns reveal hand strength far more than single actions.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from packages.common.types import ActionType


_STREETS = ("preflop", "flop", "turn", "river")
_AGGRESSIVE = {ActionType.BET, ActionType.RAISE, ActionType.ALL_IN}
_PASSIVE = {ActionType.CHECK, ActionType.CALL}


@dataclass
class StreetSequence:
    """Actions taken across streets in a single hand."""

    preflop: ActionType | None = None
    flop: ActionType | None = None
    turn: ActionType | None = None
    river: ActionType | None = None
    was_aggressor: bool = False  # was last preflop aggressor

    # Track whether opponent faced a bet on each street
    flop_facing_bet: bool = False
    turn_facing_bet: bool = False
    river_facing_bet: bool = False

    def action_on(self, street: str) -> ActionType | None:
        return getattr(self, street, None)

    def is_aggressive_on(self, street: str) -> bool:
        a = self.action_on(street)
        return a is not None and a in _AGGRESSIVE

    def is_passive_on(self, street: str) -> bool:
        a = self.action_on(street)
        return a is not None and a in _PASSIVE

    def is_check_on(self, street: str) -> bool:
        return self.action_on(street) == ActionType.CHECK


@dataclass
class PatternSignature:
    """Multi-street pattern frequencies."""

    triple_barrel_freq: float = 0.0
    double_barrel_freq: float = 0.0
    give_up_freq: float = 0.0
    check_raise_freq: float = 0.0
    probe_bet_freq: float = 0.0
    float_freq: float = 0.0
    delayed_cbet_freq: float = 0.0
    samples: int = 0


class StreetPatternTracker:
    """Tracks multi-street behavioral patterns for one opponent.

    Usage::

        tracker = StreetPatternTracker()
        tracker.start_hand(is_aggressor=True)
        tracker.record_street_action("preflop", ActionType.RAISE)
        tracker.record_street_action("flop", ActionType.BET)
        tracker.record_street_action("turn", ActionType.CHECK)
        tracker.end_hand()
        sig = tracker.get_signature()
    """

    def __init__(self, window_size: int = 100) -> None:
        self._window_size = window_size
        self._history: deque[StreetSequence] = deque(maxlen=window_size)
        self._current: StreetSequence | None = None

        # Running counters for check-raise detection per street
        self._check_raise_count: dict[str, int] = {s: 0 for s in _STREETS}
        self._check_raise_opps: dict[str, int] = {s: 0 for s in _STREETS}

    # -- hand lifecycle ----------------------------------------------------

    def start_hand(self, is_aggressor: bool = False) -> None:
        """Start tracking a new hand."""
        self._current = StreetSequence(was_aggressor=is_aggressor)

    def record_street_action(
        self,
        street: str,
        action: ActionType,
        facing_bet: bool = False,
    ) -> None:
        """Record an action on a specific street.

        Only the *first* (or most significant) action per street is stored
        in the sequence.  Aggressive actions override passive ones when
        both occur on the same street (e.g., check then raise = check-raise).
        """
        if self._current is None:
            self.start_hand()
            assert self._current is not None

        street = street.lower()

        # Store facing-bet flag
        if street == "flop":
            self._current.flop_facing_bet = self._current.flop_facing_bet or facing_bet
        elif street == "turn":
            self._current.turn_facing_bet = self._current.turn_facing_bet or facing_bet
        elif street == "river":
            self._current.river_facing_bet = self._current.river_facing_bet or facing_bet

        prev = self._current.action_on(street)

        # Check-raise detection: had CHECK, now RAISE/BET while facing bet
        if prev == ActionType.CHECK and action in _AGGRESSIVE and facing_bet:
            self._check_raise_count[street] = self._check_raise_count.get(street, 0) + 1

        if prev is not None and action == ActionType.CHECK and prev == ActionType.CHECK:
            self._check_raise_opps[street] = self._check_raise_opps.get(street, 0) + 1

        # Store the most significant action (aggressive > passive > fold)
        if prev is None or (action in _AGGRESSIVE and prev not in _AGGRESSIVE):
            setattr(self._current, street, action)

    def end_hand(self) -> None:
        """Finalize current hand and add to history."""
        if self._current is not None:
            self._history.append(self._current)
            self._current = None

    # -- analysis ----------------------------------------------------------

    def get_signature(self) -> PatternSignature:
        """Compute current pattern signature from tracked hands."""
        n = len(self._history)
        if n == 0:
            return PatternSignature(samples=0)

        triple = 0
        double = 0
        give_up = 0
        probe = 0
        float_count = 0
        delayed_cbet = 0

        # Denominators
        agg_flop = 0  # hands where player was aggressive on flop
        agg_flop_turn = 0  # hands with flop aggression going to turn
        called_flop = 0  # hands where player called on flop
        agg_preflop_check_flop = 0  # aggressor who checked flop

        for seq in self._history:
            f_agg = seq.is_aggressive_on("flop")
            t_agg = seq.is_aggressive_on("turn")
            r_agg = seq.is_aggressive_on("river")
            f_check = seq.is_check_on("flop")
            f_call = seq.action_on("flop") == ActionType.CALL

            if f_agg:
                agg_flop += 1

            # Double barrel: bet flop + bet turn
            if f_agg and t_agg:
                double += 1
                agg_flop_turn += 1
                # Triple barrel: bet flop + bet turn + bet river
                if r_agg:
                    triple += 1

            # Give up: bet flop then check turn or check river
            if f_agg and seq.is_check_on("turn"):
                give_up += 1
            elif f_agg and t_agg and seq.is_check_on("river"):
                give_up += 1

            # Probe bet: opponent (non-aggressor) bets when aggressor checks
            if not seq.was_aggressor and t_agg and seq.turn_facing_bet is False:
                probe += 1

            # Float: call flop, bet turn when checked to
            if f_call and t_agg and not seq.turn_facing_bet:
                float_count += 1
                called_flop += 1
            elif f_call:
                called_flop += 1

            # Delayed c-bet: aggressor checks flop, bets turn
            if seq.was_aggressor and f_check and t_agg:
                delayed_cbet += 1
                agg_preflop_check_flop += 1
            elif seq.was_aggressor and f_check:
                agg_preflop_check_flop += 1

        # Check-raise freq (aggregate across streets)
        cr_total = sum(self._check_raise_count.values())
        cr_opps = sum(self._check_raise_opps.values())

        sig = PatternSignature(samples=n)
        sig.double_barrel_freq = double / max(1, agg_flop)
        sig.triple_barrel_freq = triple / max(1, agg_flop_turn) if agg_flop_turn else (triple / max(1, agg_flop))
        sig.give_up_freq = give_up / max(1, agg_flop)
        sig.check_raise_freq = cr_total / max(1, cr_opps)
        sig.probe_bet_freq = probe / max(1, n)
        sig.float_freq = float_count / max(1, called_flop)
        sig.delayed_cbet_freq = delayed_cbet / max(1, agg_preflop_check_flop)

        return sig

    def predict_continuation(self, street: str) -> float:
        """Given aggression on previous street, probability of continuing.

        E.g., ``predict_continuation("turn")`` returns probability of
        betting the turn given a bet on the flop.
        """
        street = street.lower()
        if street == "flop":
            # No "previous street" for flop; return overall flop aggression rate
            total = sum(1 for s in self._history if s.action_on("flop") is not None)
            agg = sum(1 for s in self._history if s.is_aggressive_on("flop"))
            return agg / max(1, total)

        prev_map = {"turn": "flop", "river": "turn"}
        prev_street = prev_map.get(street)
        if prev_street is None:
            return 0.5

        agg_prev = [s for s in self._history if s.is_aggressive_on(prev_street)]
        if not agg_prev:
            return 0.5

        continued = sum(1 for s in agg_prev if s.is_aggressive_on(street))
        return continued / len(agg_prev)

    def exploitability_score(self) -> float:
        """How predictable is multi-street play?

        0 = balanced/unpredictable, 1 = very predictable.

        Measures how far barrel frequencies deviate from balanced ranges.
        A balanced player double-barrels ~55% and triple-barrels ~40%.
        """
        sig = self.get_signature()
        if sig.samples < 10:
            return 0.0

        # Balanced reference points (approximate GTO)
        balanced = {
            "double_barrel": 0.55,
            "triple_barrel": 0.40,
            "give_up": 0.30,
            "check_raise": 0.08,
        }

        deviations = [
            abs(sig.double_barrel_freq - balanced["double_barrel"]),
            abs(sig.triple_barrel_freq - balanced["triple_barrel"]),
            abs(sig.give_up_freq - balanced["give_up"]),
            abs(sig.check_raise_freq - balanced["check_raise"]),
        ]

        # Average deviation, normalised so 0.3 avg deviation = 1.0
        avg_dev = sum(deviations) / len(deviations)
        return min(1.0, avg_dev / 0.30)

    def reset(self) -> None:
        """Clear all tracked data."""
        self._history.clear()
        self._current = None
        self._check_raise_count = {s: 0 for s in _STREETS}
        self._check_raise_opps = {s: 0 for s in _STREETS}

    @property
    def hands_tracked(self) -> int:
        """Number of completed hands in the window."""
        return len(self._history)
