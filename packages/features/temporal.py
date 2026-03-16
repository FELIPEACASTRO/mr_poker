from __future__ import annotations

from dataclasses import dataclass

from packages.common.types import ActionType
from packages.engine.models import HandState


@dataclass(frozen=True)
class TemporalFeatures:
    """Features derived from action history within the current hand."""

    actions_this_street: int
    total_actions: int
    raises_this_street: int
    total_raises: int
    bets_this_street: int
    checks_this_street: int
    street_aggression: float
    hand_aggression: float
    last_action_type: str
    facing_raise_count: int


_AGGRESSIVE_ACTIONS = {
    ActionType.BET,
    ActionType.RAISE,
    ActionType.ALL_IN,
}

_PASSIVE_ACTIONS = {
    ActionType.CALL,
    ActionType.CHECK,
}


def derive_temporal_features(state: HandState) -> TemporalFeatures:
    """Extract temporal features from the action sequence."""
    current_street = state.street

    total_actions = 0
    total_raises = 0
    street_actions = 0
    street_raises = 0
    street_bets = 0
    street_checks = 0
    street_aggressive = 0
    street_passive = 0
    total_aggressive = 0
    total_passive = 0
    last_action_type = ""
    facing_raise_count = 0

    for action in state.actions:
        at = action.action_type
        if at in {ActionType.POST_SMALL_BLIND, ActionType.POST_BIG_BLIND}:
            continue

        total_actions += 1
        if at in _AGGRESSIVE_ACTIONS:
            total_aggressive += 1
            total_raises += 1
        elif at in _PASSIVE_ACTIONS:
            total_passive += 1

        if action.street == current_street:
            street_actions += 1
            last_action_type = at.value
            if at in _AGGRESSIVE_ACTIONS:
                street_aggressive += 1
                street_raises += 1
                if at == ActionType.BET:
                    street_bets += 1
            elif at == ActionType.CHECK:
                street_checks += 1
            elif at in _PASSIVE_ACTIONS:
                street_passive += 1

        if at in {ActionType.RAISE, ActionType.ALL_IN}:
            facing_raise_count += 1

    street_total_decisions = street_aggressive + street_passive
    total_decisions = total_aggressive + total_passive
    street_aggression = (
        street_aggressive / street_total_decisions if street_total_decisions > 0 else 0.0
    )
    hand_aggression = (
        total_aggressive / total_decisions if total_decisions > 0 else 0.0
    )

    return TemporalFeatures(
        actions_this_street=street_actions,
        total_actions=total_actions,
        raises_this_street=street_raises,
        total_raises=total_raises,
        bets_this_street=street_bets,
        checks_this_street=street_checks,
        street_aggression=round(street_aggression, 4),
        hand_aggression=round(hand_aggression, 4),
        last_action_type=last_action_type,
        facing_raise_count=facing_raise_count,
    )
