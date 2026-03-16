"""Responsible Gaming Module — Detects risk behaviors in the user.

Inverts tilt detection and fatigue modeling to protect the user instead
of exploiting the opponent. Detects: excessive sessions, tilt-driven play,
loss chasing, stake escalation, and emotional distress patterns.

Reference: Vietnam Journal of Computer Science (Feb 2026) - ML for problem detection
Reference: docs/106_Analise_Estrategica_Avancada.md Section 2.2.2
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# Enums & data classes
# ---------------------------------------------------------------------------


class RiskLevel(Enum):
    """Problem-gambling risk classification."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskIndicators:
    """Snapshot of all risk dimensions for the user."""

    session_duration_minutes: float
    hands_played: int
    loss_streak: int
    max_loss_streak: int
    session_pnl_bb: float
    tilt_score: float          # 0-1
    fatigue_score: float       # 0-1
    stake_escalation: float    # 0-1 (increase in average bet size)
    decision_speed_deviation: float  # vs personal baseline
    loss_chasing_score: float  # 0-1
    break_compliance: float    # 0-1 (did user take suggested breaks?)


@dataclass
class RiskAlert:
    """An alert generated when risk thresholds are exceeded."""

    level: RiskLevel
    message: str
    suggestion: str
    indicators: RiskIndicators
    timestamp: float


@dataclass
class SessionLimits:
    """Configurable session limits for responsible gaming."""

    max_duration_minutes: float = 120.0
    max_hands: int = 500
    max_loss_bb: float = -50.0
    break_interval_minutes: float = 60.0
    cooldown_after_tilt_minutes: float = 15.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    """Logistic sigmoid, clamped to avoid overflow."""
    x = max(-20.0, min(20.0, x))
    return 1.0 / (1.0 + math.exp(-x))


# Risk score component weights
_RISK_WEIGHTS = {
    "session_duration": 0.10,
    "loss_streak": 0.15,
    "session_pnl": 0.15,
    "tilt": 0.20,
    "fatigue": 0.10,
    "stake_escalation": 0.15,
    "loss_chasing": 0.10,
    "break_compliance": 0.05,
}


# ---------------------------------------------------------------------------
# ResponsibleGamingMonitor
# ---------------------------------------------------------------------------


class ResponsibleGamingMonitor:
    """Monitors user behavior for signs of problem gambling.

    Uses inverted versions of TiltDetector and FatigueModel signals
    to protect the user.  Generates alerts and suggestions when risk
    thresholds are exceeded.

    Parameters
    ----------
    limits:
        Session limits to enforce.  Defaults to conservative values.
    seed:
        Random seed (reserved for future probabilistic extensions).
    """

    def __init__(self, limits: SessionLimits | None = None, seed: int = 42) -> None:
        self._limits = limits or SessionLimits()
        self._seed = seed

        # Session tracking
        self._session_start: float = time.time()
        self._hands_played: int = 0
        self._session_pnl_bb: float = 0.0

        # Loss streak tracking
        self._current_loss_streak: int = 0
        self._max_loss_streak: int = 0

        # Tilt tracking (simplified internal version, mirrors TiltDetector)
        self._recent_bad_beats: int = 0
        self._aggressive_actions_recent: int = 0
        self._total_actions_recent: int = 0
        self._action_count: int = 0

        # Fatigue tracking (simplified internal version, mirrors FatigueModel)
        self._decision_times: list[float] = []
        self._baseline_decision_time: float = 0.0
        self._baseline_samples: int = 0

        # Stake tracking
        self._stakes_history: list[float] = []
        self._initial_stake: float = 0.0

        # Loss chasing: track bet sizes after losses
        self._post_loss_bets: list[float] = []
        self._post_win_bets: list[float] = []
        self._last_hand_won: bool | None = None

        # Break compliance
        self._breaks_suggested: int = 0
        self._breaks_taken: int = 0
        self._last_break_suggestion: float = 0.0
        self._last_break_taken: float = 0.0

        # Alert history
        self._alerts: list[RiskAlert] = []

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_user_action(
        self,
        action_type: str,
        street: str,
        bet_fraction: float = 0.0,
    ) -> RiskAlert | None:
        """Record a user action and check for risk patterns.

        Parameters
        ----------
        action_type:
            Action string (fold, check, call, bet, raise, all_in).
        street:
            Street string (preflop, flop, turn, river).
        bet_fraction:
            Bet/raise size as fraction of pot.

        Returns
        -------
        A :class:`RiskAlert` if any threshold is exceeded, else ``None``.
        """
        self._action_count += 1
        self._total_actions_recent += 1

        aggressive = action_type in ("bet", "raise", "all_in")
        if aggressive:
            self._aggressive_actions_recent += 1

        # Track post-loss / post-win bet sizing for loss chasing
        if aggressive and bet_fraction > 0.0:
            if self._last_hand_won is False:
                self._post_loss_bets.append(bet_fraction)
            elif self._last_hand_won is True:
                self._post_win_bets.append(bet_fraction)

        # Track decision speed (use current timestamp as a proxy —
        # real implementation would receive elapsed time)
        now = time.time()
        if self._decision_times:
            gap_ms = (now - self._decision_times[-1]) * 1000.0
            # We store raw timestamps; decision_speed_deviation uses them
        self._decision_times.append(now)

        # Periodic tilt-state check
        return self._maybe_alert()

    def record_hand_result(
        self,
        profit_bb: float,
        was_bad_beat: bool = False,
    ) -> RiskAlert | None:
        """Record hand result and check for loss chasing / tilt.

        Parameters
        ----------
        profit_bb:
            Profit or loss in big blinds for this hand.
        was_bad_beat:
            Whether the hand was a bad beat (lost with strong hand).

        Returns
        -------
        A :class:`RiskAlert` if thresholds are exceeded, else ``None``.
        """
        self._hands_played += 1
        self._session_pnl_bb += profit_bb

        won = profit_bb > 0.0
        self._last_hand_won = won

        if won:
            self._current_loss_streak = 0
        else:
            if profit_bb < 0.0:
                self._current_loss_streak += 1
                if was_bad_beat:
                    self._current_loss_streak += 1  # bad beats count double
                self._max_loss_streak = max(
                    self._max_loss_streak, self._current_loss_streak
                )

        if was_bad_beat:
            self._recent_bad_beats += 1

        return self._maybe_alert()

    def record_stake_change(self, new_stake_bb: float) -> RiskAlert | None:
        """Record when user changes stakes (detects escalation after losses).

        Parameters
        ----------
        new_stake_bb:
            The new stake level in big blinds.

        Returns
        -------
        A :class:`RiskAlert` if stake escalation is detected after losses.
        """
        if not self._stakes_history:
            self._initial_stake = new_stake_bb
        self._stakes_history.append(new_stake_bb)

        # Immediate alert if escalating while losing
        if (
            self._session_pnl_bb < 0.0
            and len(self._stakes_history) >= 2
            and new_stake_bb > self._stakes_history[-2]
        ):
            indicators = self.get_indicators()
            alert = RiskAlert(
                level=RiskLevel.HIGH,
                message="Stake escalation detected while losing. "
                        "This is a common loss-chasing pattern.",
                suggestion="Consider returning to your previous stake level "
                           "and taking a short break before continuing.",
                indicators=indicators,
                timestamp=time.time(),
            )
            self._alerts.append(alert)
            return alert

        return self._maybe_alert()

    def record_break(self) -> None:
        """Record that the user took a break (for compliance tracking)."""
        self._breaks_taken += 1
        self._last_break_taken = time.time()

    def record_decision_time(self, time_ms: float) -> None:
        """Record an explicit decision time measurement in milliseconds.

        Used to build a personal baseline and detect deviation.
        """
        # Build baseline from first 20 measurements
        if self._baseline_samples < 20:
            self._baseline_samples += 1
            self._baseline_decision_time = (
                self._baseline_decision_time * (self._baseline_samples - 1)
                + time_ms
            ) / self._baseline_samples

    # ------------------------------------------------------------------
    # Session limit checks
    # ------------------------------------------------------------------

    def check_session_limits(self) -> RiskAlert | None:
        """Check if session limits have been exceeded.

        Returns
        -------
        A :class:`RiskAlert` if any limit is exceeded, else ``None``.
        """
        duration = self._session_duration_minutes()
        limits = self._limits

        # Duration limit
        if duration >= limits.max_duration_minutes:
            return self._create_alert(
                RiskLevel.HIGH,
                f"Session duration ({duration:.0f} min) exceeds limit "
                f"({limits.max_duration_minutes:.0f} min).",
                "End this session and take a break. Rest is essential "
                "for maintaining decision quality.",
            )

        # Hands limit
        if self._hands_played >= limits.max_hands:
            return self._create_alert(
                RiskLevel.HIGH,
                f"Hands played ({self._hands_played}) exceeds limit "
                f"({limits.max_hands}).",
                "You have played a large number of hands. Consider "
                "stopping to review your session.",
            )

        # Loss limit
        if self._session_pnl_bb <= limits.max_loss_bb:
            return self._create_alert(
                RiskLevel.CRITICAL,
                f"Session loss ({self._session_pnl_bb:+.1f} BB) exceeds "
                f"stop-loss ({limits.max_loss_bb:+.1f} BB).",
                "Stop-loss limit reached. End this session immediately. "
                "Playing through large losses leads to poor decisions.",
            )

        # Break interval
        minutes_since_break = self._minutes_since_last_break()
        if minutes_since_break >= limits.break_interval_minutes:
            self._breaks_suggested += 1
            self._last_break_suggestion = time.time()
            return self._create_alert(
                RiskLevel.MODERATE,
                f"You have been playing for {minutes_since_break:.0f} minutes "
                f"without a break.",
                "Take a 5-10 minute break. Stand up, stretch, hydrate. "
                "Regular breaks improve decision quality.",
            )

        return None

    # ------------------------------------------------------------------
    # Risk scoring
    # ------------------------------------------------------------------

    def compute_risk_score(self) -> float:
        """Composite risk score 0-1.

        Combines all risk dimensions using weighted scoring similar to
        the TiltDetector's CTI (Composite Tilt Indicator), but applied
        to user-protection signals instead.
        """
        indicators = self.get_indicators()

        # Normalize each component to [0, 1]
        norm_duration = min(1.0, indicators.session_duration_minutes /
                           self._limits.max_duration_minutes)
        norm_streak = min(1.0, indicators.loss_streak / 8.0)
        norm_pnl = min(1.0, max(0.0, -indicators.session_pnl_bb) /
                       abs(self._limits.max_loss_bb))
        norm_tilt = indicators.tilt_score
        norm_fatigue = indicators.fatigue_score
        norm_escalation = indicators.stake_escalation
        norm_chasing = indicators.loss_chasing_score
        norm_compliance = 1.0 - indicators.break_compliance  # invert: low compliance = high risk

        weighted_sum = (
            _RISK_WEIGHTS["session_duration"] * norm_duration
            + _RISK_WEIGHTS["loss_streak"] * norm_streak
            + _RISK_WEIGHTS["session_pnl"] * norm_pnl
            + _RISK_WEIGHTS["tilt"] * norm_tilt
            + _RISK_WEIGHTS["fatigue"] * norm_fatigue
            + _RISK_WEIGHTS["stake_escalation"] * norm_escalation
            + _RISK_WEIGHTS["loss_chasing"] * norm_chasing
            + _RISK_WEIGHTS["break_compliance"] * norm_compliance
        )

        # Sigmoid mapping: weighted_sum ~0.3 maps to risk ~0.5
        raw = 5.0 * (weighted_sum - 0.3)
        return _sigmoid(raw)

    def get_risk_level(self) -> RiskLevel:
        """Current risk level based on composite score.

        Thresholds:
        - score < 0.25 : LOW
        - score < 0.50 : MODERATE
        - score < 0.75 : HIGH
        - score >= 0.75: CRITICAL
        """
        if self._hands_played < 2:
            return RiskLevel.LOW

        score = self.compute_risk_score()
        if score < 0.25:
            return RiskLevel.LOW
        if score < 0.50:
            return RiskLevel.MODERATE
        if score < 0.75:
            return RiskLevel.HIGH
        return RiskLevel.CRITICAL

    def get_indicators(self) -> RiskIndicators:
        """Current risk indicators snapshot.

        Returns
        -------
        A :class:`RiskIndicators` with all current values.
        """
        return RiskIndicators(
            session_duration_minutes=self._session_duration_minutes(),
            hands_played=self._hands_played,
            loss_streak=self._current_loss_streak,
            max_loss_streak=self._max_loss_streak,
            session_pnl_bb=self._session_pnl_bb,
            tilt_score=self._compute_tilt_score(),
            fatigue_score=self._compute_fatigue_score(),
            stake_escalation=self._compute_stake_escalation(),
            decision_speed_deviation=self._compute_speed_deviation(),
            loss_chasing_score=self._compute_loss_chasing_score(),
            break_compliance=self._compute_break_compliance(),
        )

    # ------------------------------------------------------------------
    # Suggestions
    # ------------------------------------------------------------------

    def suggest_break(self) -> str | None:
        """Returns break suggestion if warranted, None otherwise.

        A break is suggested when:
        - The break interval has elapsed since the last break.
        - Tilt score exceeds 0.4.
        - Fatigue score exceeds 0.5.
        - A loss streak of 5+ hands is active.
        """
        minutes_since = self._minutes_since_last_break()

        if minutes_since >= self._limits.break_interval_minutes:
            self._breaks_suggested += 1
            self._last_break_suggestion = time.time()
            return (
                f"You have been playing for {minutes_since:.0f} minutes "
                f"without a break. Take 5-10 minutes to reset."
            )

        tilt = self._compute_tilt_score()
        if tilt > 0.4:
            return (
                "Your play patterns suggest emotional stress. "
                f"Consider a {self._limits.cooldown_after_tilt_minutes:.0f}-minute "
                "cooldown before continuing."
            )

        fatigue = self._compute_fatigue_score()
        if fatigue > 0.5:
            return (
                "Signs of fatigue detected (session length, decision patterns). "
                "A short break will help maintain focus."
            )

        if self._current_loss_streak >= 5:
            return (
                f"You are on a {self._current_loss_streak}-hand losing streak. "
                "Take a moment to reset mentally before the next hand."
            )

        return None

    # ------------------------------------------------------------------
    # Session summary
    # ------------------------------------------------------------------

    def session_summary(self) -> dict:
        """End-of-session responsible gaming report.

        Returns a dict with session stats, risk assessment, and
        recommendations for future play.
        """
        indicators = self.get_indicators()
        risk_level = self.get_risk_level()
        risk_score = self.compute_risk_score()

        recommendations: list[str] = []

        if indicators.session_duration_minutes > self._limits.max_duration_minutes:
            recommendations.append(
                "Session exceeded time limit. Set a timer for your next session."
            )
        if indicators.tilt_score > 0.3:
            recommendations.append(
                "Tilt patterns detected. Practice bankroll management and "
                "emotional awareness exercises."
            )
        if indicators.loss_chasing_score > 0.3:
            recommendations.append(
                "Loss chasing behavior detected. Set strict stop-loss limits "
                "and commit to honoring them."
            )
        if indicators.stake_escalation > 0.3:
            recommendations.append(
                "Stake escalation detected. Stick to one stake level per session."
            )
        if indicators.fatigue_score > 0.5:
            recommendations.append(
                "Fatigue indicators elevated. Shorten your sessions or add "
                "more breaks."
            )
        if indicators.break_compliance < 0.5 and self._breaks_suggested > 0:
            recommendations.append(
                "Break compliance was low. Commit to taking breaks when suggested."
            )
        if not recommendations:
            recommendations.append(
                "Good session discipline. Keep maintaining your current habits."
            )

        return {
            "session_duration_minutes": indicators.session_duration_minutes,
            "hands_played": indicators.hands_played,
            "session_pnl_bb": indicators.session_pnl_bb,
            "max_loss_streak": indicators.max_loss_streak,
            "risk_score": round(risk_score, 3),
            "risk_level": risk_level.value,
            "indicators": {
                "tilt_score": round(indicators.tilt_score, 3),
                "fatigue_score": round(indicators.fatigue_score, 3),
                "stake_escalation": round(indicators.stake_escalation, 3),
                "loss_chasing_score": round(indicators.loss_chasing_score, 3),
                "break_compliance": round(indicators.break_compliance, 3),
            },
            "alerts_generated": len(self._alerts),
            "breaks_suggested": self._breaks_suggested,
            "breaks_taken": self._breaks_taken,
            "recommendations": recommendations,
        }

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset for new session."""
        self._session_start = time.time()
        self._hands_played = 0
        self._session_pnl_bb = 0.0
        self._current_loss_streak = 0
        self._max_loss_streak = 0
        self._recent_bad_beats = 0
        self._aggressive_actions_recent = 0
        self._total_actions_recent = 0
        self._action_count = 0
        self._decision_times.clear()
        # Keep baseline decision time across sessions
        self._stakes_history.clear()
        self._initial_stake = 0.0
        self._post_loss_bets.clear()
        self._post_win_bets.clear()
        self._last_hand_won = None
        self._breaks_suggested = 0
        self._breaks_taken = 0
        self._last_break_suggestion = 0.0
        self._last_break_taken = 0.0
        self._alerts.clear()

    # ------------------------------------------------------------------
    # Internal: component scores
    # ------------------------------------------------------------------

    def _session_duration_minutes(self) -> float:
        """Minutes elapsed since session start."""
        return (time.time() - self._session_start) / 60.0

    def _compute_tilt_score(self) -> float:
        """Internal tilt score 0-1, mirroring TiltDetector logic.

        Uses aggression ratio increase, bad-beat count, and loss streak
        as proxies for emotional state.
        """
        if self._total_actions_recent < 5:
            return 0.0

        # Aggression ratio (baseline ~0.3 for typical play)
        agg_ratio = self._aggressive_actions_recent / max(1, self._total_actions_recent)
        agg_excess = max(0.0, agg_ratio - 0.30) / 0.30  # normalized

        # Bad beat impact (each bad beat adds ~0.15)
        bad_beat_component = min(1.0, self._recent_bad_beats * 0.15)

        # Loss streak impact
        streak_component = min(1.0, self._current_loss_streak / 6.0)

        # P&L impact (losing amplifies tilt)
        pnl_component = min(1.0, max(0.0, -self._session_pnl_bb) / 40.0)

        raw = (
            0.30 * agg_excess
            + 0.25 * bad_beat_component
            + 0.25 * streak_component
            + 0.20 * pnl_component
        )

        return min(1.0, max(0.0, raw))

    def _compute_fatigue_score(self) -> float:
        """Internal fatigue score 0-1, mirroring FatigueModel logic.

        Based on session duration using saturating exponential curve.
        """
        minutes = self._session_duration_minutes()
        # Duration component: 1 - exp(-minutes / 180)
        duration_score = 1.0 - math.exp(-minutes / 180.0)

        # Speed deviation adds fatigue signal
        speed_dev = self._compute_speed_deviation()
        speed_component = min(0.15, max(0.0, speed_dev) * 0.05)

        return min(1.0, duration_score + speed_component)

    def _compute_stake_escalation(self) -> float:
        """Stake escalation score 0-1.

        Measures how much the user has increased stakes relative to
        their starting level.  Escalation while losing is especially
        risky.
        """
        if len(self._stakes_history) < 2 or self._initial_stake <= 0:
            return 0.0

        current_stake = self._stakes_history[-1]
        ratio = current_stake / self._initial_stake

        if ratio <= 1.0:
            return 0.0  # no escalation

        # Raw escalation: doubling stakes = 0.5, tripling = 0.75
        raw = min(1.0, (ratio - 1.0) / 2.0)

        # Amplify if losing
        if self._session_pnl_bb < 0.0:
            loss_multiplier = min(2.0, 1.0 + abs(self._session_pnl_bb) / 50.0)
            raw = min(1.0, raw * loss_multiplier)

        return raw

    def _compute_speed_deviation(self) -> float:
        """Decision speed deviation from personal baseline.

        Positive = slower than baseline (fatigue signal).
        Negative = faster than baseline (possible tilt / impatience).
        Returns 0.0 if insufficient data.
        """
        if self._baseline_decision_time <= 0 or self._baseline_samples < 10:
            return 0.0

        # Use time gaps between recorded decision timestamps as proxy
        if len(self._decision_times) < 10:
            return 0.0

        recent_gaps = []
        for i in range(max(0, len(self._decision_times) - 20),
                       len(self._decision_times) - 1):
            gap = (self._decision_times[i + 1] - self._decision_times[i]) * 1000.0
            recent_gaps.append(gap)

        if not recent_gaps:
            return 0.0

        avg_recent = sum(recent_gaps) / len(recent_gaps)
        deviation = (avg_recent - self._baseline_decision_time) / max(
            1.0, self._baseline_decision_time
        )
        return deviation

    def _compute_loss_chasing_score(self) -> float:
        """Loss chasing score 0-1.

        Compares average bet size after losses vs after wins.
        A higher ratio indicates loss chasing behavior.
        """
        if len(self._post_loss_bets) < 3 or len(self._post_win_bets) < 3:
            return 0.0

        avg_post_loss = sum(self._post_loss_bets) / len(self._post_loss_bets)
        avg_post_win = sum(self._post_win_bets) / len(self._post_win_bets)

        if avg_post_win <= 0.0:
            return 0.0

        ratio = avg_post_loss / avg_post_win

        if ratio <= 1.0:
            return 0.0  # not chasing

        # ratio 1.5 = mild chasing (0.33), ratio 2.0 = significant (0.67)
        return min(1.0, (ratio - 1.0) / 1.5)

    def _compute_break_compliance(self) -> float:
        """Break compliance score 0-1 (1 = fully compliant).

        Ratio of breaks taken to breaks suggested.  Returns 1.0 if
        no breaks have been suggested yet.
        """
        if self._breaks_suggested == 0:
            return 1.0
        return min(1.0, self._breaks_taken / self._breaks_suggested)

    def _minutes_since_last_break(self) -> float:
        """Minutes since the last break (or session start)."""
        reference = max(self._session_start, self._last_break_taken)
        return (time.time() - reference) / 60.0

    # ------------------------------------------------------------------
    # Internal: alert generation
    # ------------------------------------------------------------------

    def _maybe_alert(self) -> RiskAlert | None:
        """Check all risk dimensions and return an alert if warranted.

        Only generates an alert if the risk level is MODERATE or above,
        and avoids spamming by requiring at least 5 hands between alerts.
        """
        # Anti-spam: at least 5 hands between alerts
        if self._alerts and self._hands_played > 0:
            last_alert_hands = len(self._alerts)  # rough proxy
            if last_alert_hands > 0 and self._hands_played % 5 != 0:
                return None

        # Check session limits first (most actionable)
        limit_alert = self.check_session_limits()
        if limit_alert is not None:
            return limit_alert

        # Check composite risk
        risk_level = self.get_risk_level()
        if risk_level == RiskLevel.LOW:
            return None

        indicators = self.get_indicators()

        if risk_level == RiskLevel.CRITICAL:
            alert = RiskAlert(
                level=RiskLevel.CRITICAL,
                message="Multiple risk indicators are at critical levels. "
                        "Strong recommendation to stop playing.",
                suggestion="End this session now. Take at least a 30-minute break. "
                           "Consider reviewing your session with a clear head.",
                indicators=indicators,
                timestamp=time.time(),
            )
        elif risk_level == RiskLevel.HIGH:
            # Identify the dominant risk factor
            dominant = self._dominant_risk_factor(indicators)
            alert = RiskAlert(
                level=RiskLevel.HIGH,
                message=f"Elevated risk detected: {dominant}.",
                suggestion=self._suggestion_for_factor(dominant),
                indicators=indicators,
                timestamp=time.time(),
            )
        else:  # MODERATE
            alert = RiskAlert(
                level=RiskLevel.MODERATE,
                message="Moderate risk indicators detected. Stay aware.",
                suggestion="Monitor your mental state. Consider taking a "
                           "short break if you notice frustration or fatigue.",
                indicators=indicators,
                timestamp=time.time(),
            )

        self._alerts.append(alert)
        return alert

    def _create_alert(
        self, level: RiskLevel, message: str, suggestion: str
    ) -> RiskAlert:
        """Helper to create and store an alert."""
        indicators = self.get_indicators()
        alert = RiskAlert(
            level=level,
            message=message,
            suggestion=suggestion,
            indicators=indicators,
            timestamp=time.time(),
        )
        self._alerts.append(alert)
        return alert

    @staticmethod
    def _dominant_risk_factor(indicators: RiskIndicators) -> str:
        """Identify the most elevated risk factor."""
        factors = {
            "tilt": indicators.tilt_score,
            "fatigue": indicators.fatigue_score,
            "loss chasing": indicators.loss_chasing_score,
            "stake escalation": indicators.stake_escalation,
            "extended session": min(1.0, indicators.session_duration_minutes / 120.0),
            "loss streak": min(1.0, indicators.loss_streak / 6.0),
        }
        return max(factors, key=factors.get)  # type: ignore[arg-type]

    @staticmethod
    def _suggestion_for_factor(factor: str) -> str:
        """Return a specific suggestion for the dominant risk factor."""
        suggestions = {
            "tilt": "Take a 15-minute cooldown break. Practice deep breathing. "
                    "Do not make any stake changes until you feel calm.",
            "fatigue": "End the session or take a 10-minute break. Physical "
                       "movement and hydration help restore focus.",
            "loss chasing": "Your bet sizing has increased after losses. "
                           "Return to your standard sizing and consider "
                           "stopping for the day.",
            "stake escalation": "Moving up in stakes while losing is risky. "
                               "Return to your previous stake level.",
            "extended session": "Long sessions degrade decision quality. "
                               "Take a break or end the session.",
            "loss streak": "A losing streak can affect your judgment. "
                          "Take a short break to reset mentally.",
        }
        return suggestions.get(
            factor,
            "Take a break and reassess your state before continuing.",
        )

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        risk = self.get_risk_level()
        return (
            f"ResponsibleGamingMonitor(hands={self._hands_played}, "
            f"risk={risk.value}, "
            f"pnl={self._session_pnl_bb:+.1f}BB)"
        )
