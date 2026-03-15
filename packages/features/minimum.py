from __future__ import annotations

from dataclasses import dataclass

from packages.engine.models import HandState, PlayerState


@dataclass
class MinimumDecisionFeatures:
    street: str
    pot: int
    to_call: int
    player_stack: int
    board_size: int
    acting_seat: int | None
    is_button: bool
    current_bet: int


def derive_minimum_features(state: HandState, player: PlayerState) -> MinimumDecisionFeatures:
    return MinimumDecisionFeatures(
        street=state.street.value,
        pot=state.pot,
        to_call=state.to_call if state.to_call else max(0, state.current_bet - player.invested_this_round),
        player_stack=player.stack,
        board_size=len(state.board),
        acting_seat=state.acting_seat,
        is_button=player.seat == state.button_seat,
        current_bet=state.current_bet,
    )
