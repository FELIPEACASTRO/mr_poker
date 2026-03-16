"""Opponent Modeling — Player classification and dynamic exploitation.

Classifies opponents into archetypes (TAG, LAG, Nit, Maniac, Fish)
based on observed actions, and adjusts strategy accordingly.

Based on:
- Suspicion-Agent (2023): Theory of Mind for poker
- Beyond GTO: Hybrid GTO + exploitation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from packages.common.types import ActionType


@dataclass
class PlayerStats:
    """Running statistics for an opponent."""
    total_hands: int = 0
    voluntary_put_in_pot: int = 0   # VPIP
    preflop_raises: int = 0         # PFR
    three_bets: int = 0             # 3-bet
    fold_to_three_bet: int = 0
    three_bet_opportunities: int = 0
    cbet_made: int = 0              # C-bet
    cbet_opportunities: int = 0
    fold_to_cbet: int = 0
    cbet_faced: int = 0
    total_aggressive_actions: int = 0  # bets + raises
    total_passive_actions: int = 0     # checks + calls
    total_folds: int = 0
    went_to_showdown: int = 0       # WTSD
    won_at_showdown: int = 0        # W$SD
    total_postflop_actions: int = 0
    check_raises: int = 0
    donk_bets: int = 0
    overbet_count: int = 0
    # Per-street aggression
    flop_aggression: int = 0
    flop_actions: int = 0
    turn_aggression: int = 0
    turn_actions: int = 0
    river_aggression: int = 0
    river_actions: int = 0

    @property
    def vpip(self) -> float:
        return self.voluntary_put_in_pot / max(1, self.total_hands)

    @property
    def pfr(self) -> float:
        return self.preflop_raises / max(1, self.total_hands)

    @property
    def three_bet_pct(self) -> float:
        return self.three_bets / max(1, self.three_bet_opportunities)

    @property
    def fold_to_3bet_pct(self) -> float:
        return self.fold_to_three_bet / max(1, self.three_bet_opportunities)

    @property
    def cbet_pct(self) -> float:
        return self.cbet_made / max(1, self.cbet_opportunities)

    @property
    def fold_to_cbet_pct(self) -> float:
        return self.fold_to_cbet / max(1, self.cbet_faced)

    @property
    def aggression_factor(self) -> float:
        return self.total_aggressive_actions / max(1, self.total_passive_actions)

    @property
    def wtsd(self) -> float:
        return self.went_to_showdown / max(1, self.total_hands)

    @property
    def wsd(self) -> float:
        return self.won_at_showdown / max(1, self.went_to_showdown)

    @property
    def fold_pct(self) -> float:
        total = self.total_aggressive_actions + self.total_passive_actions + self.total_folds
        return self.total_folds / max(1, total)


# Player archetypes
ARCHETYPES = {
    "nit": "Tight-passive. Plays few hands, rarely bluffs. Fold to aggression.",
    "tag": "Tight-aggressive. Standard winning style. Hard to exploit.",
    "lag": "Loose-aggressive. Wide range, lots of aggression. Bluffs often.",
    "maniac": "Hyper-aggressive. Over-bets, over-bluffs. Call down wider.",
    "fish": "Loose-passive. Calls too much, rarely raises. Value bet thin.",
    "whale": "Very loose, very passive. Massive leaks. Max value extraction.",
    "rock": "Ultra-tight. Only plays premium. Fold everything vs their aggression.",
    "unknown": "Not enough data. Play GTO baseline.",
}


def classify_player(stats: PlayerStats) -> str:
    """Classify player into an archetype based on observed stats."""
    if stats.total_hands < 10:
        return "unknown"

    vpip = stats.vpip
    pfr = stats.pfr
    af = stats.aggression_factor
    fold_pct = stats.fold_pct

    # Ultra-tight players
    if vpip < 0.15:
        return "rock" if af < 2.0 else "nit"

    # Tight players (VPIP 15-28%)
    if vpip < 0.28:
        if af >= 2.5 and pfr >= 0.15:
            return "tag"
        return "nit"

    # Moderate players (VPIP 28-40%)
    if vpip < 0.40:
        if af >= 3.0:
            return "lag"
        if af >= 2.0:
            return "tag"
        return "fish"

    # Loose players (VPIP 40%+)
    if af >= 3.5:
        return "maniac"
    if af >= 2.0:
        return "lag"
    if fold_pct < 0.25:
        return "whale"
    return "fish"


def exploitation_adjustments(archetype: str) -> dict[str, float]:
    """Return strategy adjustments based on opponent archetype.

    Returns multipliers for different actions:
    - bluff_freq: how often to bluff (1.0 = GTO frequency)
    - value_bet_freq: how often to value bet thin
    - fold_freq: how often to fold marginal hands
    - raise_freq: how often to raise/3-bet
    - call_freq: how often to call down
    """
    adjustments = {
        "unknown": {
            "bluff_freq": 1.0, "value_bet_freq": 1.0,
            "fold_freq": 1.0, "raise_freq": 1.0, "call_freq": 1.0,
        },
        "nit": {
            "bluff_freq": 1.5,    # Bluff more — they fold a lot
            "value_bet_freq": 0.7, # Don't thin value bet — they have it
            "fold_freq": 1.3,      # Fold more vs their aggression
            "raise_freq": 0.7,     # Don't raise light
            "call_freq": 0.6,      # Fold more when they bet
        },
        "rock": {
            "bluff_freq": 1.8,
            "value_bet_freq": 0.5,
            "fold_freq": 1.5,
            "raise_freq": 0.5,
            "call_freq": 0.4,
        },
        "tag": {
            "bluff_freq": 0.9,    # Slightly less bluffing — they're balanced
            "value_bet_freq": 1.0,
            "fold_freq": 1.0,
            "raise_freq": 1.0,
            "call_freq": 1.0,
        },
        "lag": {
            "bluff_freq": 0.6,    # Bluff less — they call/raise light
            "value_bet_freq": 1.3, # Value bet thinner
            "fold_freq": 0.8,      # Don't over-fold
            "raise_freq": 1.2,     # Re-raise more
            "call_freq": 1.3,      # Call down wider
        },
        "maniac": {
            "bluff_freq": 0.3,    # Rarely bluff — let them bluff
            "value_bet_freq": 1.5, # Value bet very thin
            "fold_freq": 0.5,      # Call much more
            "raise_freq": 1.5,     # Raise for value
            "call_freq": 1.6,      # Call down with marginal hands
        },
        "fish": {
            "bluff_freq": 0.5,    # Don't bluff calling stations
            "value_bet_freq": 1.4, # Value bet thin — they'll call
            "fold_freq": 0.9,
            "raise_freq": 1.1,
            "call_freq": 1.0,
        },
        "whale": {
            "bluff_freq": 0.2,    # Never bluff — they never fold
            "value_bet_freq": 1.6, # Value bet everything
            "fold_freq": 0.7,
            "raise_freq": 1.3,
            "call_freq": 1.2,
        },
    }
    return adjustments.get(archetype, adjustments["unknown"])


@dataclass
class OpponentTracker:
    """Tracks statistics for all opponents across hands."""
    stats: dict[int, PlayerStats] = field(default_factory=dict)

    def get_or_create(self, seat: int) -> PlayerStats:
        if seat not in self.stats:
            self.stats[seat] = PlayerStats()
        return self.stats[seat]

    def record_action(
        self,
        seat: int,
        action: ActionType,
        *,
        street: str = "pre_flop",
        is_preflop_raise: bool = False,
        facing_bet: bool = False,
        pot_size: int = 0,
        bet_amount: int = 0,
    ) -> None:
        """Record an opponent action for statistics tracking."""
        s = self.get_or_create(seat)

        if action in {ActionType.BET, ActionType.RAISE, ActionType.ALL_IN}:
            s.total_aggressive_actions += 1
        elif action in {ActionType.CHECK, ActionType.CALL}:
            s.total_passive_actions += 1
        elif action == ActionType.FOLD:
            s.total_folds += 1

        if street == "pre_flop":
            if action in {ActionType.CALL, ActionType.BET, ActionType.RAISE, ActionType.ALL_IN}:
                s.voluntary_put_in_pot += 1
            if action in {ActionType.RAISE, ActionType.BET, ActionType.ALL_IN}:
                s.preflop_raises += 1
        else:
            s.total_postflop_actions += 1
            if street == "flop":
                s.flop_actions += 1
                if action in {ActionType.BET, ActionType.RAISE}:
                    s.flop_aggression += 1
            elif street == "turn":
                s.turn_actions += 1
                if action in {ActionType.BET, ActionType.RAISE}:
                    s.turn_aggression += 1
            elif street == "river":
                s.river_actions += 1
                if action in {ActionType.BET, ActionType.RAISE}:
                    s.river_aggression += 1

        # Overbet detection
        if action in {ActionType.BET, ActionType.RAISE} and pot_size > 0:
            if bet_amount > pot_size:
                s.overbet_count += 1

    def record_hand_end(self, seat: int, *, went_to_showdown: bool = False, won: bool = False) -> None:
        s = self.get_or_create(seat)
        s.total_hands += 1
        if went_to_showdown:
            s.went_to_showdown += 1
            if won:
                s.won_at_showdown += 1

    def classify(self, seat: int) -> str:
        if seat not in self.stats:
            return "unknown"
        return classify_player(self.stats[seat])

    def get_adjustments(self, seat: int) -> dict[str, float]:
        archetype = self.classify(seat)
        return exploitation_adjustments(archetype)

    def compute_exploit_blend(self, seat: int) -> float:
        """Compute how much to deviate from GTO based on confidence.

        Returns 0.0 (pure GTO) to 0.7 (max exploitation).
        Requires minimum 20 hands before exploiting.
        """
        if seat not in self.stats:
            return 0.0
        s = self.stats[seat]
        if s.total_hands < 20:
            return 0.0

        archetype = classify_player(s)
        if archetype in ("unknown", "tag"):
            return 0.0  # Don't exploit TAG or unknown

        # Confidence grows with sample size, capped at 0.7
        confidence = min(0.7, (s.total_hands - 20) / 200.0)

        # Scale by how exploitable the archetype is
        exploit_scale = {
            "nit": 0.5, "rock": 0.6, "lag": 0.4,
            "maniac": 0.7, "fish": 0.6, "whale": 0.8,
        }
        return confidence * exploit_scale.get(archetype, 0.0)
