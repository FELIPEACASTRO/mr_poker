"""Tilt Detection & Emotional State Modeling.

Detects tilt in opponents based on behavioral indicators from scientific
research (Palomaki 2013, Cortisol/PNAS, Princeton Prospect Theory).

Tracks: VPIP delta, raise freq after bad beats, fold-to-3bet changes,
loss streaks, overbet spikes, session P&L.
"""

from __future__ import annotations

import enum
import math
from collections import deque
from dataclasses import dataclass, field

from packages.common.types import ActionType


# ---------------------------------------------------------------------------
# Enums & data classes
# ---------------------------------------------------------------------------

class TiltState(enum.Enum):
    """Emotional state classification."""

    NORMAL = "normal"
    MILD_TILT = "mild_tilt"
    FULL_TILT = "full_tilt"
    STEAMING = "steaming"


@dataclass
class HandResult:
    """Result of a single hand for P&L tracking."""

    profit_bb: float = 0.0  # profit in big blinds
    was_bad_beat: bool = False  # lost with strong hand
    went_to_showdown: bool = False
    won: bool = False


@dataclass
class TiltIndicators:
    """Raw tilt indicator values.

    Each field captures one dimension of tilt.  Positive values indicate
    behaviour shifting *toward* tilt (looser, more aggressive, less
    disciplined).
    """

    vpip_delta: float = 0.0  # recent VPIP - baseline VPIP
    raise_freq_delta: float = 0.0  # recent raise freq - baseline
    fold_to_3bet_delta: float = 0.0  # baseline f23b - recent f23b (inverted)
    loss_streak: int = 0  # consecutive losses
    overbet_spike: float = 0.0  # recent overbet rate / baseline
    session_pnl_bb: float = 0.0  # session profit/loss in BB


# ---------------------------------------------------------------------------
# Internal action record
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class _ActionRecord:
    """Lightweight internal record of one observed action."""

    action: ActionType
    street: str
    bet_fraction: float


# ---------------------------------------------------------------------------
# CTI weights (Composite Tilt Indicator)
# ---------------------------------------------------------------------------

_WEIGHTS = {
    "vpip_delta": 0.25,
    "raise_freq_delta": 0.20,
    "fold_to_3bet_delta": 0.15,
    "loss_streak": 0.20,
    "overbet_spike": 0.10,
    "session_pnl": 0.10,
}


def _sigmoid(x: float) -> float:
    """Standard logistic sigmoid, clamped to avoid overflow."""
    x = max(-20.0, min(20.0, x))
    return 1.0 / (1.0 + math.exp(-x))


# ---------------------------------------------------------------------------
# Helpers for computing rates from action deques
# ---------------------------------------------------------------------------

def _vpip_rate(actions: deque[_ActionRecord]) -> float:
    """Fraction of preflop actions that are voluntary put-in-pot."""
    preflop = [a for a in actions if a.street == "preflop"]
    if not preflop:
        return 0.0
    vpip_actions = {ActionType.CALL, ActionType.BET, ActionType.RAISE, ActionType.ALL_IN}
    return sum(1 for a in preflop if a.action in vpip_actions) / len(preflop)


def _raise_rate(actions: deque[_ActionRecord]) -> float:
    """Fraction of all actions that are raises / bets / all-in."""
    if not actions:
        return 0.0
    raise_actions = {ActionType.BET, ActionType.RAISE, ActionType.ALL_IN}
    return sum(1 for a in actions if a.action in raise_actions) / len(actions)


def _overbet_rate(actions: deque[_ActionRecord]) -> float:
    """Fraction of bets/raises that are overbets (bet_fraction > 1.0)."""
    bets = [a for a in actions if a.action in {ActionType.BET, ActionType.RAISE, ActionType.ALL_IN}]
    if not bets:
        return 0.0
    return sum(1 for a in bets if a.bet_fraction > 1.0) / len(bets)


def _fold_to_3bet_rate(actions: deque[_ActionRecord]) -> float:
    """Approximation: fraction of preflop folds among all preflop actions.

    A true fold-to-3bet rate requires context about facing 3-bets.  As a
    proxy we use preflop fold frequency — tilted players fold *less*
    preflop, so a drop in this value signals tilt.
    """
    preflop = [a for a in actions if a.street == "preflop"]
    if not preflop:
        return 0.0
    return sum(1 for a in preflop if a.action == ActionType.FOLD) / len(preflop)


# ---------------------------------------------------------------------------
# TiltDetector
# ---------------------------------------------------------------------------

class TiltDetector:
    """Detects tilt state from behavioural patterns.

    Uses a sliding window of recent hands (default 10) compared against
    a baseline window (default 100) to detect behavioural shifts that
    indicate emotional state changes.

    Scientific basis:
    - Palomaki et al. (2013): +15% VPIP after bad beats
    - PNAS Cortisol study: stress increases risk-seeking
    - Princeton 4.9M hands: loss aversion at 100BB reference point
    - Eil & Lien (2013): "getting even" effect when losing
    """

    def __init__(
        self,
        recent_window: int = 10,
        baseline_window: int = 100,
        seed: int = 42,
    ) -> None:
        self._recent_window = max(1, recent_window)
        self._baseline_window = max(1, baseline_window)
        self._seed = seed

        # Action tracking — recent and baseline
        self._recent_actions: deque[_ActionRecord] = deque(maxlen=self._recent_window * 4)
        self._baseline_actions: deque[_ActionRecord] = deque(maxlen=self._baseline_window * 4)

        # Hand results
        self._recent_results: deque[HandResult] = deque(maxlen=self._recent_window)
        self._baseline_results: deque[HandResult] = deque(maxlen=self._baseline_window)

        # Session-level accumulators
        self._session_pnl_bb: float = 0.0
        self._current_loss_streak: int = 0
        self._total_hands: int = 0
        self._total_actions: int = 0

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_action(
        self,
        action: ActionType,
        street: str = "preflop",
        bet_fraction: float = 0.0,
    ) -> None:
        """Record an observed action.

        Parameters
        ----------
        action:
            The action type observed.
        street:
            One of "preflop", "flop", "turn", "river".
        bet_fraction:
            Bet/raise size as a fraction of the pot (e.g. 1.5 = 150% pot).
            Only meaningful for BET / RAISE / ALL_IN.
        """
        rec = _ActionRecord(action=action, street=street, bet_fraction=bet_fraction)
        self._recent_actions.append(rec)
        self._baseline_actions.append(rec)
        self._total_actions += 1

    def record_hand_result(self, result: HandResult) -> None:
        """Record hand outcome for P&L and streak tracking."""
        self._recent_results.append(result)
        self._baseline_results.append(result)
        self._session_pnl_bb += result.profit_bb
        self._total_hands += 1

        # Update loss streak
        if result.won:
            self._current_loss_streak = 0
        else:
            # Only count hands that actually cost chips (not free walks)
            if result.profit_bb < 0.0 or result.went_to_showdown:
                self._current_loss_streak += 1

        # Bad beats amplify streak impact
        if result.was_bad_beat:
            self._current_loss_streak += 1  # counts double

    # ------------------------------------------------------------------
    # Indicator computation
    # ------------------------------------------------------------------

    def compute_indicators(self) -> TiltIndicators:
        """Compute all tilt indicators from recent vs baseline windows.

        Returns a :class:`TiltIndicators` dataclass with the raw delta
        values.  Each positive value means the player is shifting toward
        more tilted behaviour.
        """
        # VPIP delta
        recent_vpip = _vpip_rate(self._recent_actions)
        baseline_vpip = _vpip_rate(self._baseline_actions)
        vpip_delta = recent_vpip - baseline_vpip

        # Raise frequency delta
        recent_raise = _raise_rate(self._recent_actions)
        baseline_raise = _raise_rate(self._baseline_actions)
        raise_freq_delta = recent_raise - baseline_raise

        # Fold-to-3bet delta (inverted — lower fold rate = more tilt)
        recent_f23b = _fold_to_3bet_rate(self._recent_actions)
        baseline_f23b = _fold_to_3bet_rate(self._baseline_actions)
        fold_to_3bet_delta = baseline_f23b - recent_f23b  # positive = tilted

        # Loss streak
        loss_streak = self._current_loss_streak

        # Overbet spike
        recent_overbet = _overbet_rate(self._recent_actions)
        baseline_overbet = _overbet_rate(self._baseline_actions)
        if baseline_overbet > 0.01:
            overbet_spike = recent_overbet / baseline_overbet
        else:
            # No baseline overbets — if recent has some, that is a spike
            overbet_spike = recent_overbet * 10.0

        # Session P&L (negative = losing)
        session_pnl_bb = self._session_pnl_bb

        return TiltIndicators(
            vpip_delta=vpip_delta,
            raise_freq_delta=raise_freq_delta,
            fold_to_3bet_delta=fold_to_3bet_delta,
            loss_streak=loss_streak,
            overbet_spike=overbet_spike,
            session_pnl_bb=session_pnl_bb,
        )

    def composite_tilt_score(self) -> float:
        """Composite Tilt Indicator (CTI): 0.0 (calm) to 1.0 (full tilt).

        Weighted combination of all normalised indicators passed through a
        sigmoid to produce a 0-1 score.

        Indicator normalisation:
        - vpip_delta: raw (typically -0.3 to +0.3)
        - raise_freq_delta: raw (similar range)
        - fold_to_3bet_delta: raw
        - loss_streak: divided by 5 (so 5 losses ~ 1.0 contribution)
        - overbet_spike: (spike - 1.0) clamped >= 0
        - session_pnl: -pnl / 50 (losing 50BB ~ 1.0 contribution)

        The normalised indicators are multiplied by their weights, summed,
        then scaled and passed through sigmoid.
        """
        ind = self.compute_indicators()

        # Normalise each indicator to roughly [0, 1] scale when tilting
        norm_vpip = max(0.0, ind.vpip_delta) / 0.15  # +15% VPIP = 1.0
        norm_raise = max(0.0, ind.raise_freq_delta) / 0.15
        norm_f23b = max(0.0, ind.fold_to_3bet_delta) / 0.15
        norm_streak = ind.loss_streak / 5.0
        norm_overbet = max(0.0, ind.overbet_spike - 1.0) / 2.0
        norm_pnl = max(0.0, -ind.session_pnl_bb) / 50.0  # losing 50BB = 1.0

        weighted_sum = (
            _WEIGHTS["vpip_delta"] * norm_vpip
            + _WEIGHTS["raise_freq_delta"] * norm_raise
            + _WEIGHTS["fold_to_3bet_delta"] * norm_f23b
            + _WEIGHTS["loss_streak"] * norm_streak
            + _WEIGHTS["overbet_spike"] * norm_overbet
            + _WEIGHTS["session_pnl"] * norm_pnl
        )

        # Scale so that a weighted_sum of ~0.5 maps to CTI ~0.5
        # sigmoid(0) = 0.5, so we shift: sigmoid(4*(ws - 0.25))
        raw = 4.0 * (weighted_sum - 0.25)
        return _sigmoid(raw)

    # ------------------------------------------------------------------
    # Tilt classification
    # ------------------------------------------------------------------

    def detect_tilt(self) -> TiltState:
        """Classify current tilt state based on CTI score.

        Thresholds:
        - CTI < 0.25 : NORMAL
        - CTI < 0.50 : MILD_TILT
        - CTI < 0.75 : FULL_TILT
        - CTI >= 0.75: STEAMING
        """
        if self._total_hands < 2:
            return TiltState.NORMAL

        cti = self.composite_tilt_score()
        if cti < 0.25:
            return TiltState.NORMAL
        if cti < 0.50:
            return TiltState.MILD_TILT
        if cti < 0.75:
            return TiltState.FULL_TILT
        return TiltState.STEAMING

    # ------------------------------------------------------------------
    # Exploitation helpers
    # ------------------------------------------------------------------

    def exploit_adjustment(self) -> float:
        """How much to increase exploit_blend (0.0 to 0.3 additional).

        When an opponent is tilting they become significantly more
        exploitable.  This value can be added to the base exploit_blend
        from :class:`OpponentTracker.compute_exploit_blend`.
        """
        cti = self.composite_tilt_score()
        # Map CTI to 0.0-0.3 with a slight dead-zone for NORMAL
        if cti < 0.25:
            return 0.0
        # Linear ramp from 0.0 at CTI=0.25 to 0.3 at CTI=1.0
        return min(0.3, 0.4 * (cti - 0.25))

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def session_pnl(self) -> float:
        """Current session P&L in BB."""
        return self._session_pnl_bb

    @property
    def hands_tracked(self) -> int:
        """Total number of hand results recorded."""
        return self._total_hands

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all tracking data."""
        self._recent_actions.clear()
        self._baseline_actions.clear()
        self._recent_results.clear()
        self._baseline_results.clear()
        self._session_pnl_bb = 0.0
        self._current_loss_streak = 0
        self._total_hands = 0
        self._total_actions = 0

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        state = self.detect_tilt()
        return (
            f"TiltDetector(hands={self._total_hands}, "
            f"state={state.value}, "
            f"pnl={self._session_pnl_bb:+.1f}BB)"
        )
