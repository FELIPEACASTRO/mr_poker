"""Kelly Criterion — Optimal bankroll management for poker.

Implements the Kelly Criterion and its conservative variants for
sizing bets relative to bankroll based on estimated edge and odds.

Reference: Kelly (1956) "A New Interpretation of Information Rate"
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

logger = logging.getLogger(__name__)


def kelly_fraction(edge: float, odds: float) -> float:
    """Compute the Kelly fraction for a single bet.

    The Kelly fraction maximizes the expected logarithm of wealth
    (geometric growth rate).

    Args:
        edge: Expected edge (probability of winning - probability of losing).
               E.g., if win_prob=0.55, edge = 0.55 - 0.45 = 0.10
        odds: Payout odds (net profit / amount risked).
               E.g., odds=1.0 means even money, odds=2.0 means 2:1.

    Returns:
        Fraction of bankroll to bet (0.0 to 1.0, clamped).
    """
    if odds <= 0:
        return 0.0
    if edge <= 0:
        return 0.0

    # Kelly formula: f* = edge / odds
    # More precisely: f* = (p * (b + 1) - 1) / b
    # where p = win probability, b = odds
    # Simplified: f* = edge / odds when edge = p - q and odds = b
    fraction = edge / odds
    return max(0.0, min(1.0, fraction))


def kelly_from_probability(win_prob: float, odds: float) -> float:
    """Compute Kelly fraction from win probability and odds.

    Args:
        win_prob: Probability of winning (0.0 to 1.0).
        odds: Net payout odds (e.g., 1.0 for even money).

    Returns:
        Kelly fraction (0.0 to 1.0).
    """
    if win_prob <= 0 or win_prob >= 1 or odds <= 0:
        return 0.0

    # f* = (p * (b + 1) - 1) / b
    fraction = (win_prob * (odds + 1) - 1) / odds
    return max(0.0, min(1.0, fraction))


def half_kelly(win_prob: float, odds: float) -> float:
    """Half-Kelly: more conservative, reduces variance by ~75%.

    Half-Kelly achieves ~75% of the growth rate of full Kelly
    while cutting variance in half. Widely used by professional
    gamblers and fund managers.
    """
    return kelly_from_probability(win_prob, odds) * 0.5


def quarter_kelly(win_prob: float, odds: float) -> float:
    """Quarter-Kelly: very conservative, for high-variance situations.

    Quarter-Kelly is appropriate when:
    - Edge estimates are uncertain
    - Bankroll is limited
    - Variance tolerance is low (tournament play)
    """
    return kelly_from_probability(win_prob, odds) * 0.25


@dataclass
class BankrollManager:
    """Bankroll management using Kelly Criterion variants.

    Tracks bankroll and recommends bet sizes based on estimated edge.
    """

    bankroll: float
    kelly_multiplier: float = 0.5  # Default: half-Kelly
    min_bet: float = 0.0
    max_bet_fraction: float = 0.25  # Never bet more than 25% of bankroll

    def recommended_bet(self, win_prob: float, pot_odds: float) -> float:
        """Compute recommended bet size.

        Args:
            win_prob: Estimated probability of winning the hand.
            pot_odds: Odds offered by the pot (pot / amount_to_call).

        Returns:
            Recommended bet amount (in bankroll units).
        """
        if self.bankroll <= 0:
            return 0.0

        full_kelly = kelly_from_probability(win_prob, pot_odds)
        adjusted = full_kelly * self.kelly_multiplier
        bet_fraction = min(adjusted, self.max_bet_fraction)
        bet = self.bankroll * bet_fraction

        return max(self.min_bet, bet)

    def update_bankroll(self, result: float) -> None:
        """Update bankroll after a hand result."""
        self.bankroll += result

    def risk_of_ruin(self, edge: float, std_dev: float, target_multiple: float = 0.0) -> float:
        """Estimate probability of going bust.

        Uses the classic risk-of-ruin formula for repeated bets.

        Args:
            edge: Expected win rate per hand (in BB).
            std_dev: Standard deviation per hand (in BB).
            target_multiple: Target bankroll multiple (0 = bust).

        Returns:
            Estimated probability of ruin (0.0 to 1.0).
        """
        if std_dev <= 0 or edge <= 0:
            return 1.0 if edge <= 0 else 0.0

        # Classic formula: RoR = exp(-2 * edge * bankroll / std_dev^2)
        exponent = -2.0 * edge * self.bankroll / (std_dev ** 2)
        return min(1.0, math.exp(exponent))

    def buyins_remaining(self, buyin_size: float) -> float:
        """How many buy-ins remain in the bankroll."""
        if buyin_size <= 0:
            return float("inf")
        return self.bankroll / buyin_size

    def should_move_down(self, buyin_size: float, min_buyins: int = 20) -> bool:
        """Whether to move down in stakes based on bankroll."""
        return self.buyins_remaining(buyin_size) < min_buyins

    def optimal_stake(self, buyins_required: int = 30) -> float:
        """Calculate the maximum buy-in that maintains proper bankroll management."""
        if buyins_required <= 0:
            return 0.0
        return self.bankroll / buyins_required


class KellyCriterion:
    """Bankroll management via Kelly Criterion.

    Provides optimal bet sizing, play/no-play decisions, and
    risk-of-ruin estimates using the Kelly Criterion framework.

    Uses a fractional Kelly approach (default quarter-Kelly) for
    conservative bankroll management suitable for poker.
    """

    def __init__(self, bankroll: float, fraction: float = 0.25) -> None:
        """Initialize Kelly Criterion calculator.

        Args:
            bankroll: Current bankroll size.
            fraction: Kelly fraction (0.25 = quarter Kelly, conservative).
        """
        self.bankroll = bankroll
        self.fraction = fraction

    def optimal_bet(self, win_prob: float, pot_odds: float) -> float:
        """Calculate optimal bet size as fraction of bankroll.

        Kelly formula: f* = (bp - q) / b
        where b = pot odds, p = win prob, q = 1-p

        Args:
            win_prob: Probability of winning (0 to 1).
            pot_odds: Net payout odds (pot / cost to call).

        Returns:
            Optimal bet amount (fraction of bankroll * Kelly fraction).
        """
        if win_prob <= 0 or win_prob >= 1 or pot_odds <= 0:
            return 0.0

        p = win_prob
        q = 1.0 - p
        b = pot_odds

        # Kelly formula: f* = (bp - q) / b
        kelly_f = (b * p - q) / b

        if kelly_f <= 0:
            return 0.0

        # Apply fractional Kelly and bankroll
        bet = self.bankroll * kelly_f * self.fraction
        return max(0.0, bet)

    def should_play(self, win_prob: float, pot_odds: float) -> bool:
        """Returns True if the hand has positive expected Kelly growth.

        A hand is worth playing if the Kelly fraction is positive,
        meaning the expected value is positive given the odds.
        """
        if win_prob <= 0 or win_prob >= 1 or pot_odds <= 0:
            return False

        p = win_prob
        q = 1.0 - p
        b = pot_odds

        kelly_f = (b * p - q) / b
        return kelly_f > 0

    def update_bankroll(self, result: float) -> None:
        """Update bankroll after a hand.

        Args:
            result: Net profit/loss from the hand.
        """
        self.bankroll += result

    def risk_of_ruin(self, target_multiple: float = 2.0) -> float:
        """Estimate probability of going broke before reaching target.

        Uses the simplified risk-of-ruin formula:
            RoR = (1 - edge)^(bankroll / unit) approximately
            RoR = ((1-f)/f)^n for Kelly betting

        For fractional Kelly with fraction f applied to edge e:
            RoR ~ (q/p)^(B/unit) where B = bankroll

        Args:
            target_multiple: Target bankroll multiple (e.g., 2.0 = double).

        Returns:
            Estimated probability of ruin (0.0 to 1.0).
        """
        if self.bankroll <= 0:
            return 1.0
        if target_multiple <= 1.0:
            return 0.0

        # For fractional Kelly, risk of ruin before reaching target:
        # RoR = 1 - (1 - (1/target)^(2*edge_factor))
        # Simplified: using Kelly fraction as proxy for edge
        edge_factor = self.fraction  # conservative proxy

        if edge_factor <= 0:
            return 1.0

        # Classic formula approximation
        exponent = 2.0 * edge_factor * math.log(target_multiple)
        ror = math.exp(-exponent * self.bankroll / 100.0)
        return min(1.0, max(0.0, ror))
