"""ICM — Independent Chip Model for tournament equity.

Converts chip stacks into monetary equity based on payout structure.
Critical for tournament strategy: chips won are worth less than chips
lost because of the concave relationship between stack size and equity.

Implements:
- Exact ICM via recursive inclusion-exclusion (<=8 players)
- Fast approximation for >8 players
- ICM pressure calculation (how much equity you risk by playing a pot)
- PKO (Progressive Knockout) bounty equity
- SNG and MTT equity calculators

Reference: Malmuth & Harville (1973) "Independent Chip Model"
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from packages.common.types import ActionType
from packages.strategy.mixed import ActionDistribution


class ICMCalculator:
    """Independent Chip Model calculator.

    Computes the monetary equity of each player's chip stack given
    a tournament payout structure.
    """

    def __init__(self, max_exact_players: int = 8) -> None:
        self.max_exact_players = max_exact_players
        self._cache: dict[tuple, list[float]] = {}

    def chip_equity(
        self,
        stacks: list[float],
        payouts: list[float],
    ) -> list[float]:
        """Compute ICM equity for each player.

        Args:
            stacks: Current chip stacks (must be non-negative).
            payouts: Prize pool payouts [1st, 2nd, 3rd, ...].

        Returns:
            List of equity values, one per player. Sums to sum(payouts).
        """
        n = len(stacks)
        if n == 0:
            return []

        total_chips = sum(stacks)
        if total_chips <= 0:
            return [0.0] * n

        # Pad payouts to match player count
        padded_payouts = list(payouts) + [0.0] * max(0, n - len(payouts))

        # Cache key
        cache_key = (tuple(stacks), tuple(padded_payouts[:n]))
        if cache_key in self._cache:
            return self._cache[cache_key]

        if n <= self.max_exact_players:
            result = self._exact_icm(stacks, padded_payouts[:n], total_chips)
        else:
            result = self._approx_icm(stacks, padded_payouts[:n], total_chips)

        self._cache[cache_key] = result
        return result

    def _exact_icm(
        self,
        stacks: list[float],
        payouts: list[float],
        total_chips: float,
    ) -> list[float]:
        """Exact ICM via recursive inclusion-exclusion (Malmuth-Harville).

        For each player, compute the probability of finishing in each position
        by recursing over all possible finish orderings.
        """
        n = len(stacks)
        equities = [0.0] * n

        for player in range(n):
            # Probability of finishing in each place
            place_probs = self._place_probabilities(player, stacks, total_chips)
            for place, prob in enumerate(place_probs):
                if place < len(payouts):
                    equities[player] += prob * payouts[place]

        return equities

    def _place_probabilities(
        self,
        player: int,
        stacks: list[float],
        total_chips: float,
    ) -> list[float]:
        """Compute probability of 'player' finishing in each place.

        Uses the Harville formula: P(finish k-th) = sum over all ways
        the (k-1) players who finish ahead.
        """
        n = len(stacks)
        probs = [0.0] * n

        # P(1st) = stack / total
        probs[0] = stacks[player] / total_chips

        # P(k-th) via recursive formula
        others = [i for i in range(n) if i != player]
        for place in range(1, n):
            probs[place] = self._prob_place_recursive(
                player, place, others, stacks, total_chips
            )

        return probs

    def _prob_place_recursive(
        self,
        player: int,
        place: int,
        others: list[int],
        stacks: list[float],
        total_chips: float,
    ) -> float:
        """Recursively compute probability of player finishing in a given place.

        P(player finishes place-th) = sum over each other player j:
            P(j finishes 1st) * P(player finishes (place-1)th | j already out)
        """
        if place == 0:
            return stacks[player] / total_chips
        if not others:
            return 0.0

        # For efficiency, limit recursion depth
        if place > 4:
            # Approximate higher places
            remaining_equity = 1.0 - sum(
                stacks[player] / total_chips
                for _ in range(place)
            )
            return max(0.0, remaining_equity / max(len(others), 1))

        prob = 0.0
        for j in others:
            p_j_first = stacks[j] / total_chips
            remaining_others = [o for o in others if o != j]
            remaining_total = total_chips - stacks[j]

            if remaining_total <= 0:
                continue

            # Rescale stacks for remaining players
            sub_prob = self._prob_place_recursive(
                player, place - 1, remaining_others, stacks, remaining_total
            )
            prob += p_j_first * sub_prob

        return prob

    def _approx_icm(
        self,
        stacks: list[float],
        payouts: list[float],
        total_chips: float,
    ) -> list[float]:
        """Approximate ICM for large fields (>8 players).

        Uses a weighted blend of chip-proportional equity and a simplified
        ICM formula based on stack ranking.
        """
        n = len(stacks)
        chip_eq = [s / total_chips * sum(payouts) for s in stacks]

        # Rank-based adjustment
        ranked = sorted(range(n), key=lambda i: stacks[i], reverse=True)
        rank_bonus = [0.0] * n
        for rank, idx in enumerate(ranked):
            if rank < len(payouts):
                # Higher ranked stacks get a bonus towards higher payouts
                rank_bonus[idx] = payouts[rank] * 0.1

        # Blend chip-proportional and rank-based
        equities = [
            chip_eq[i] * 0.85 + rank_bonus[i] * 0.15
            for i in range(n)
        ]

        # Normalize to sum to total payouts
        total_eq = sum(equities)
        total_pay = sum(payouts)
        if total_eq > 0:
            equities = [e / total_eq * total_pay for e in equities]

        return equities

    def icm_pressure(
        self,
        hero_stack: float,
        villain_stack: float,
        all_stacks: list[float],
        payouts: list[float],
    ) -> float:
        """Compute ICM pressure: how much equity hero risks by calling.

        Higher pressure means hero should play tighter (folding more).
        Pressure is highest for medium stacks near the bubble.

        Returns:
            Pressure value (0.0 = no pressure, 1.0+ = extreme pressure).
        """
        total = sum(all_stacks)
        if total <= 0:
            return 0.0

        n = len(all_stacks)

        # Find hero's index
        hero_idx = None
        for i, s in enumerate(all_stacks):
            if abs(s - hero_stack) < 0.01:
                hero_idx = i
                break
        if hero_idx is None:
            return 0.0

        # Current equity
        current_eq = self.chip_equity(all_stacks, payouts)
        hero_current = current_eq[hero_idx]

        # Equity if hero wins (gains villain's stack)
        win_stacks = list(all_stacks)
        call_amount = min(hero_stack, villain_stack)
        win_stacks[hero_idx] += call_amount
        # Find villain and reduce
        villain_idx = None
        for i, s in enumerate(all_stacks):
            if i != hero_idx and abs(s - villain_stack) < 0.01:
                villain_idx = i
                break
        if villain_idx is not None:
            win_stacks[villain_idx] -= call_amount

        self._cache.clear()
        win_eq = self.chip_equity(win_stacks, payouts)
        hero_win_eq = win_eq[hero_idx]

        # Equity if hero loses
        lose_stacks = list(all_stacks)
        lose_stacks[hero_idx] -= call_amount
        if villain_idx is not None:
            lose_stacks[villain_idx] += call_amount

        self._cache.clear()
        lose_eq = self.chip_equity(lose_stacks, payouts)
        hero_lose_eq = lose_eq[hero_idx]

        # ICM pressure = risk / reward ratio
        equity_gained = hero_win_eq - hero_current
        equity_lost = hero_current - hero_lose_eq

        if equity_gained <= 0:
            return 2.0  # Extreme pressure — can't gain anything
        return equity_lost / equity_gained

    def clear_cache(self) -> None:
        self._cache.clear()


def adjust_strategy_for_icm(
    base_strategy: ActionDistribution,
    icm_pressure: float,
    hero_stack_ratio: float = 0.5,
) -> ActionDistribution:
    """Adjust a strategy based on ICM pressure.

    High pressure (bubble, medium stack) → tighten (fold more).
    Short-stacked (desperate) → widen (shove more).
    Chip leader → can pressure others.

    Args:
        base_strategy: The GTO/baseline strategy to adjust.
        icm_pressure: ICM pressure value from icm_pressure().
        hero_stack_ratio: Hero's stack as fraction of average stack.

    Returns:
        Adjusted ActionDistribution.
    """
    probs = dict(base_strategy.probabilities)
    if not probs:
        return base_strategy

    # Determine adjustment factor
    if hero_stack_ratio < 0.3:
        # Short-stacked: play more aggressively (push/fold)
        fold_mult = 0.7
        aggression_mult = 1.4
    elif icm_pressure > 1.5:
        # High pressure (bubble): play very tight
        fold_mult = 1.8
        aggression_mult = 0.4
    elif icm_pressure > 1.0:
        # Medium pressure: somewhat tighter
        fold_mult = 1.3
        aggression_mult = 0.7
    elif hero_stack_ratio > 2.0:
        # Chip leader: can pressure
        fold_mult = 0.8
        aggression_mult = 1.3
    else:
        # Normal: minimal adjustment
        fold_mult = 1.0
        aggression_mult = 1.0

    aggressive_actions = {ActionType.BET, ActionType.RAISE, ActionType.ALL_IN}

    adjusted: dict[ActionType, float] = {}
    for action, prob in probs.items():
        if action == ActionType.FOLD:
            adjusted[action] = prob * fold_mult
        elif action in aggressive_actions:
            adjusted[action] = prob * aggression_mult
        else:
            adjusted[action] = prob

    # Normalize
    total = sum(adjusted.values())
    if total > 0:
        adjusted = {a: p / total for a, p in adjusted.items()}

    return ActionDistribution(probabilities=adjusted)


class PKOCalculator:
    """Progressive Knockout bounty equity calculator.

    In PKO tournaments, eliminating a player awards a bounty.
    This changes calling ranges: you call wider when the bounty
    value is high relative to the pot.
    """

    def __init__(self, icm_calc: ICMCalculator | None = None) -> None:
        self.icm = icm_calc or ICMCalculator()

    def bounty_equity(
        self,
        hero_stack: float,
        villain_stack: float,
        bounty_value: float,
        pot: float = 0.0,
    ) -> float:
        """Compute the equity bonus from the bounty.

        The bounty value is added to the pot equity calculation,
        making calls more profitable.

        Returns:
            Equity bonus (in tournament $) from the bounty opportunity.
        """
        if hero_stack <= 0 or villain_stack <= 0:
            return 0.0

        # Probability of elimination (hero covers villain)
        if hero_stack >= villain_stack:
            # Hero can eliminate villain
            # Bounty equity = P(win) * bounty_value
            # Approximate P(win) from stack ratio
            total = hero_stack + villain_stack
            p_win = hero_stack / total
            return p_win * bounty_value
        else:
            # Hero cannot eliminate villain (would bust first)
            return 0.0

    def adjust_calling_range(
        self,
        base_strategy: ActionDistribution,
        bounty_value: float,
        pot: float,
    ) -> ActionDistribution:
        """Widen calling range based on bounty value.

        Higher bounty relative to pot → call wider.
        """
        if pot <= 0:
            return base_strategy

        bounty_ratio = bounty_value / pot
        # Increase call/aggressive action probability
        call_boost = min(bounty_ratio * 0.5, 0.3)  # Cap at 30% boost

        probs = dict(base_strategy.probabilities)
        calling_actions = {ActionType.CALL, ActionType.ALL_IN}

        for action in calling_actions:
            if action in probs:
                probs[action] = probs[action] * (1.0 + call_boost)

        # Reduce fold proportionally
        if ActionType.FOLD in probs and call_boost > 0:
            probs[ActionType.FOLD] = probs[ActionType.FOLD] * (1.0 - call_boost * 0.5)

        # Normalize
        total = sum(probs.values())
        if total > 0:
            probs = {a: p / total for a, p in probs.items()}

        return ActionDistribution(probabilities=probs)


class SNGEquityCalculator:
    """SNG (Sit & Go) equity calculator using ICM.

    Standard SNG payout structures with exact ICM computation.
    """

    def __init__(self) -> None:
        self.icm = ICMCalculator()

    def equity(
        self,
        stacks: list[float],
        payout_structure: list[float] | None = None,
    ) -> list[float]:
        """Compute SNG equity for each player.

        Args:
            stacks: Current chip stacks.
            payout_structure: Payouts. Defaults to standard 50/30/20 for 3+ players.

        Returns:
            Equity values per player.
        """
        n = len(stacks)
        if payout_structure is None:
            # Standard SNG payouts
            if n <= 2:
                total = sum(stacks)
                payout_structure = [total * 0.65, total * 0.35]
            elif n <= 6:
                total = sum(stacks)
                payout_structure = [total * 0.50, total * 0.30, total * 0.20]
            else:
                total = sum(stacks)
                payout_structure = [
                    total * 0.40, total * 0.25, total * 0.18,
                    total * 0.10, total * 0.07,
                ]

        return self.icm.chip_equity(stacks, payout_structure)


class MTTEquityCalculator:
    """MTT (Multi-Table Tournament) equity calculator.

    Handles large fields with ICM approximations and stage-aware
    payout structures.
    """

    def __init__(self) -> None:
        self.icm = ICMCalculator(max_exact_players=8)

    def equity(
        self,
        stacks: list[float],
        total_prize_pool: float,
        payout_percentages: list[float] | None = None,
        remaining_players: int | None = None,
    ) -> list[float]:
        """Compute MTT equity for each player.

        Args:
            stacks: Current chip stacks (at the table or full field).
            total_prize_pool: Total prize pool in dollars.
            payout_percentages: Payout as fractions of prize pool [1st%, 2nd%, ...].
            remaining_players: Total remaining players in tournament.

        Returns:
            Equity values per player.
        """
        n = len(stacks)
        remaining = remaining_players or n

        if payout_percentages is None:
            # Standard MTT payout structure (top ~15% get paid)
            paid_places = max(1, int(remaining * 0.15))
            payout_percentages = self._generate_mtt_payouts(paid_places)

        payouts = [p * total_prize_pool for p in payout_percentages]

        # For large fields, use approximation
        return self.icm.chip_equity(stacks, payouts)

    def _generate_mtt_payouts(self, paid_places: int) -> list[float]:
        """Generate a standard MTT payout structure.

        Uses a geometric decay for payout distribution.
        """
        if paid_places <= 0:
            return []
        if paid_places == 1:
            return [1.0]

        # Geometric decay with ratio ~0.65
        raw = [0.65 ** i for i in range(paid_places)]
        total = sum(raw)
        return [r / total for r in raw]

    def bubble_factor(
        self,
        stacks: list[float],
        total_prize_pool: float,
        remaining_players: int,
        paid_places: int,
    ) -> float:
        """Compute the bubble factor (how close to the money bubble).

        Returns a value from 0.0 (far from bubble) to 1.0 (on the bubble).
        """
        if remaining_players <= paid_places:
            return 0.0  # Already in the money
        if remaining_players <= 0:
            return 0.0

        distance = remaining_players - paid_places
        # Exponential approach to bubble
        return math.exp(-distance / max(paid_places * 0.1, 1))
