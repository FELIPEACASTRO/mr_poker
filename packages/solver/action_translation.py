"""Action Translation / Pseudo-Harmonic Mapping.

When an opponent makes a bet size that doesn't match the solver's
abstraction grid (e.g., solver knows 0.5x pot and 1.0x pot, opponent
bets 0.75x pot), this module translates the off-grid action into a
probability distribution over the nearest abstract actions.

This preserves information that would be lost by simply rounding to
the nearest abstract action.  The translation uses pseudo-harmonic
mapping which interpolates between abstract actions proportionally
to their distance from the actual bet size.

Reference: Ganzfried & Sandholm (2013) "Action Translation in
Extensive-Form Games with Large Action Spaces", Carnegie Mellon
University.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from packages.common.types import ActionType
from packages.strategy.mixed import ActionDistribution


@dataclass
class AbstractAction:
    """An action in the solver's abstraction.

    Attributes:
        action_type: fold, check, call, bet, raise, all_in
        bet_fraction: bet/raise size as fraction of pot (0.0 for non-bet)
        label: human-readable label
    """
    action_type: ActionType = ActionType.CHECK
    bet_fraction: float = 0.0
    label: str = ""

    def __post_init__(self) -> None:
        if not self.label:
            if self.action_type in (ActionType.BET, ActionType.RAISE):
                self.label = f"{self.action_type.value}_{self.bet_fraction:.2f}x"
            else:
                self.label = self.action_type.value


@dataclass
class TranslationResult:
    """Result of translating an off-grid action.

    Attributes:
        distribution: probability distribution over abstract actions
        original_fraction: the actual bet fraction observed
        confidence: how confident we are in the translation (1.0 = exact match)
    """
    distribution: dict[str, float] = field(default_factory=dict)
    original_fraction: float = 0.0
    confidence: float = 1.0


def _pseudo_harmonic_weight(
    actual: float, abstract: float, other_abstract: float
) -> float:
    """Compute pseudo-harmonic mapping weight.

    Weight for abstract action is proportional to 1/distance to actual,
    normalized so weights sum to 1.  This gives a smooth interpolation
    that preserves pot odds relationships.
    """
    dist = abs(actual - abstract)
    if dist < 1e-10:
        return 1.0  # Exact match

    other_dist = abs(actual - other_abstract)
    if other_dist < 1e-10:
        return 0.0

    # Inverse distance weighting
    w = 1.0 / dist
    w_other = 1.0 / other_dist
    return w / (w + w_other)


class ActionTranslator:
    """Translates off-grid bet sizes to abstract action distributions.

    Given a solver's bet size grid (e.g., [0.33, 0.5, 0.75, 1.0, 1.5]),
    translates any observed bet size into a probability distribution over
    the two nearest grid points.
    """

    def __init__(
        self,
        bet_sizes: list[float] | None = None,
        include_fold: bool = True,
        include_check_call: bool = True,
    ) -> None:
        """Initialize with the solver's bet size grid.

        Args:
            bet_sizes: sorted list of bet fractions (as fraction of pot).
                Default: [0.33, 0.5, 0.75, 1.0, 1.5, 2.0]
            include_fold: include fold as an abstract action
            include_check_call: include check/call as abstract actions
        """
        self.bet_sizes = sorted(bet_sizes if bet_sizes is not None else [0.33, 0.5, 0.75, 1.0, 1.5, 2.0])
        self.include_fold = include_fold
        self.include_check_call = include_check_call
        self.abstract_actions = self._build_abstract_actions()

    def _build_abstract_actions(self) -> list[AbstractAction]:
        """Build the full list of abstract actions."""
        actions: list[AbstractAction] = []
        if self.include_fold:
            actions.append(AbstractAction(ActionType.FOLD, 0.0, "fold"))
        if self.include_check_call:
            actions.append(AbstractAction(ActionType.CHECK, 0.0, "check"))
            actions.append(AbstractAction(ActionType.CALL, 0.0, "call"))
        for size in self.bet_sizes:
            actions.append(AbstractAction(ActionType.BET, size))
        return actions

    def translate_bet(
        self,
        actual_fraction: float,
        pot_size: float = 100.0,
    ) -> TranslationResult:
        """Translate an actual bet fraction to abstract action distribution.

        Args:
            actual_fraction: actual bet size as fraction of pot
            pot_size: pot size (for context, not used in basic translation)

        Returns:
            TranslationResult with distribution over abstract bet sizes
        """
        if not self.bet_sizes:
            return TranslationResult(
                distribution={"check": 1.0},
                original_fraction=actual_fraction,
                confidence=0.0,
            )

        # Find the two nearest abstract bet sizes
        lower = None
        upper = None

        for size in self.bet_sizes:
            if size <= actual_fraction:
                lower = size
            if size >= actual_fraction and upper is None:
                upper = size

        # Exact match
        if lower is not None and abs(lower - actual_fraction) < 1e-10:
            label = f"bet_{lower:.2f}x"
            return TranslationResult(
                distribution={label: 1.0},
                original_fraction=actual_fraction,
                confidence=1.0,
            )
        if upper is not None and abs(upper - actual_fraction) < 1e-10:
            label = f"bet_{upper:.2f}x"
            return TranslationResult(
                distribution={label: 1.0},
                original_fraction=actual_fraction,
                confidence=1.0,
            )

        # Below smallest abstract size
        if lower is None and upper is not None:
            label = f"bet_{upper:.2f}x"
            dist_to_zero = actual_fraction
            dist_to_upper = upper - actual_fraction
            if self.include_check_call:
                total = dist_to_zero + dist_to_upper
                if total > 0:
                    check_w = dist_to_upper / total  # closer to 0 -> more check
                    bet_w = dist_to_zero / total
                    return TranslationResult(
                        distribution={"call": check_w, label: bet_w},
                        original_fraction=actual_fraction,
                        confidence=1.0 - dist_to_upper / max(upper, 0.01),
                    )
            return TranslationResult(
                distribution={label: 1.0},
                original_fraction=actual_fraction,
                confidence=0.5,
            )

        # Above largest abstract size
        if upper is None and lower is not None:
            label = f"bet_{lower:.2f}x"
            # Map to all-in or largest size
            return TranslationResult(
                distribution={label: 1.0},
                original_fraction=actual_fraction,
                confidence=max(0.3, 1.0 - (actual_fraction - lower) / max(lower, 0.01)),
            )

        # Between two abstract sizes: pseudo-harmonic interpolation
        if lower is not None and upper is not None:
            lower_label = f"bet_{lower:.2f}x"
            upper_label = f"bet_{upper:.2f}x"

            w_lower = _pseudo_harmonic_weight(actual_fraction, lower, upper)
            w_upper = 1.0 - w_lower

            # Confidence based on how far from grid points
            gap = upper - lower
            min_dist = min(actual_fraction - lower, upper - actual_fraction)
            confidence = 1.0 - (min_dist / gap) * 0.5 if gap > 0 else 1.0

            return TranslationResult(
                distribution={lower_label: w_lower, upper_label: w_upper},
                original_fraction=actual_fraction,
                confidence=confidence,
            )

        return TranslationResult(
            distribution={"check": 1.0},
            original_fraction=actual_fraction,
            confidence=0.0,
        )

    def translate_action(
        self,
        action: ActionType,
        bet_amount: float = 0.0,
        pot_size: float = 100.0,
    ) -> TranslationResult:
        """Translate any action (including non-bets) to abstract space.

        Args:
            action: the action type
            bet_amount: the bet/raise amount (0 for non-bet actions)
            pot_size: current pot size

        Returns:
            TranslationResult
        """
        if action == ActionType.FOLD:
            return TranslationResult(
                distribution={"fold": 1.0},
                original_fraction=0.0,
                confidence=1.0,
            )
        if action in (ActionType.CHECK, ActionType.CALL):
            label = action.value
            return TranslationResult(
                distribution={label: 1.0},
                original_fraction=0.0,
                confidence=1.0,
            )
        if action == ActionType.ALL_IN:
            if pot_size > 0:
                fraction = bet_amount / pot_size
                result = self.translate_bet(fraction, pot_size)
                # Merge all-in signal
                return result
            return TranslationResult(
                distribution={"all_in": 1.0},
                original_fraction=0.0,
                confidence=1.0,
            )

        # BET or RAISE
        if pot_size > 0 and bet_amount > 0:
            fraction = bet_amount / pot_size
            return self.translate_bet(fraction, pot_size)

        return TranslationResult(
            distribution={"check": 1.0},
            original_fraction=0.0,
            confidence=0.5,
        )

    def grid_sizes(self) -> list[float]:
        """Return the solver's bet size grid."""
        return self.bet_sizes[:]
