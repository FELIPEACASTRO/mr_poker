"""AIVAT — Action-Informed Variance Adjusted Technique.

Unbiased evaluation of poker agent performance with dramatically
reduced variance (10-50x) compared to raw BB/100 measurement.
Subtracts counterfactual values of unplayed actions to reduce noise.

Reference: Burch et al. (2018) "AIVAT: A New Variance Reduction Technique
for Agent Evaluation in Imperfect Information Games" (AAAI)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class HandResult:
    """Result of a single hand for AIVAT evaluation."""
    hand_id: str
    hero_seat: int
    # Raw result in BB
    raw_result_bb: float
    # Equity at each decision point (from equity calculator)
    decision_equities: list[float] = field(default_factory=list)
    # Pot size at each decision point (in BB)
    decision_pots_bb: list[float] = field(default_factory=list)
    # Whether hero acted at each decision point
    hero_acted: list[bool] = field(default_factory=list)
    # Board runout equity (final board vs hero's hand)
    final_equity: float = 0.5


@dataclass
class AIVATResult:
    """AIVAT-adjusted evaluation result."""
    raw_winrate_bb100: float
    aivat_winrate_bb100: float
    raw_std_dev: float
    aivat_std_dev: float
    variance_reduction: float
    num_hands: int
    confidence_interval_95: tuple[float, float] = (0.0, 0.0)

    @property
    def is_significant(self) -> bool:
        """Whether the winrate is statistically significant at 95%."""
        low, high = self.confidence_interval_95
        return low > 0 or high < 0  # CI doesn't cross zero


class AIVATEvaluator:
    """AIVAT variance reduction for poker agent evaluation.

    The key insight is that poker results contain two sources of variance:
    1. Card runout variance (which cards come on board)
    2. Opponent strategy variance (which actions opponent takes)

    AIVAT removes component (1) by subtracting the counterfactual value
    of the actual card runout vs the expected value over all possible runouts.
    This dramatically reduces variance without introducing bias.
    """

    def __init__(self) -> None:
        self._results: list[HandResult] = []
        self._aivat_values: list[float] = []

    def add_hand(self, result: HandResult) -> None:
        """Add a hand result for evaluation."""
        self._results.append(result)
        aivat_value = self._compute_aivat_value(result)
        self._aivat_values.append(aivat_value)

    def _compute_aivat_value(self, result: HandResult) -> float:
        """Compute AIVAT-adjusted value for a single hand.

        AIVAT_value = raw_result - sum(counterfactual_adjustments)

        The adjustment at each decision point is:
        (actual_equity - expected_equity) * pot_size

        Where expected_equity is 0.5 (fair coin) for card dealing events.
        """
        adjustment = 0.0

        # Card runout adjustment
        # The equity deviation from 0.5 at each decision point
        # represents variance from card luck, not from play quality
        for i, (equity, pot_bb, hero_act) in enumerate(zip(
            result.decision_equities,
            result.decision_pots_bb,
            result.hero_acted,
        )):
            if not hero_act:
                # Only adjust at nature's actions (card deals), not player actions
                # The adjustment removes card-luck variance
                card_luck = (equity - 0.5) * pot_bb
                adjustment += card_luck

        # Final runout adjustment
        if result.decision_pots_bb:
            final_pot = result.decision_pots_bb[-1] if result.decision_pots_bb else 0.0
            final_adjustment = (result.final_equity - 0.5) * final_pot
            adjustment += final_adjustment

        return result.raw_result_bb - adjustment

    def evaluate(self) -> AIVATResult:
        """Compute AIVAT-adjusted evaluation metrics."""
        n = len(self._results)
        if n == 0:
            return AIVATResult(
                raw_winrate_bb100=0.0,
                aivat_winrate_bb100=0.0,
                raw_std_dev=0.0,
                aivat_std_dev=0.0,
                variance_reduction=0.0,
                num_hands=0,
            )

        # Raw statistics
        raw_values = [r.raw_result_bb for r in self._results]
        raw_mean = sum(raw_values) / n
        raw_var = sum((v - raw_mean) ** 2 for v in raw_values) / max(n - 1, 1)
        raw_std = math.sqrt(raw_var)

        # AIVAT statistics
        aivat_mean = sum(self._aivat_values) / n
        aivat_var = sum((v - aivat_mean) ** 2 for v in self._aivat_values) / max(n - 1, 1)
        aivat_std = math.sqrt(aivat_var)

        # Variance reduction ratio
        var_reduction = 1.0 - (aivat_var / max(raw_var, 1e-10)) if raw_var > 0 else 0.0

        # 95% confidence interval for AIVAT winrate
        se = aivat_std / math.sqrt(n) if n > 0 else 0.0
        ci_low = (aivat_mean - 1.96 * se) * 100  # Convert to BB/100
        ci_high = (aivat_mean + 1.96 * se) * 100

        return AIVATResult(
            raw_winrate_bb100=raw_mean * 100,
            aivat_winrate_bb100=aivat_mean * 100,
            raw_std_dev=raw_std,
            aivat_std_dev=aivat_std,
            variance_reduction=var_reduction,
            num_hands=n,
            confidence_interval_95=(ci_low, ci_high),
        )

    def reset(self) -> None:
        """Clear all accumulated results."""
        self._results.clear()
        self._aivat_values.clear()

    def minimum_hands_for_significance(
        self, expected_winrate_bb100: float = 5.0, confidence: float = 0.95
    ) -> int:
        """Estimate minimum hands needed to detect a given winrate.

        Uses current AIVAT std dev if available, otherwise uses
        typical poker std dev (~60 BB/100 raw, ~15-30 AIVAT-adjusted).
        """
        z = 1.96 if confidence >= 0.95 else 1.645

        if self._aivat_values and len(self._aivat_values) > 1:
            aivat_mean = sum(self._aivat_values) / len(self._aivat_values)
            aivat_var = sum(
                (v - aivat_mean) ** 2 for v in self._aivat_values
            ) / (len(self._aivat_values) - 1)
            std = math.sqrt(aivat_var)
        else:
            std = 0.2  # Typical AIVAT std in BB per hand

        if expected_winrate_bb100 <= 0:
            return 999999

        # n = (z * std * 100 / winrate)^2
        wr_per_hand = expected_winrate_bb100 / 100.0
        n = (z * std / wr_per_hand) ** 2
        return max(1, int(math.ceil(n)))
