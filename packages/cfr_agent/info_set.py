"""Information set abstraction for CFR.

Maps a game state to a string key that captures what the acting player
knows (their hole cards, the board, pot, position, action history)
while abstracting away what they don't know (opponent's cards).
"""

from __future__ import annotations

from packages.engine.models import Card, HandState
from packages.features.poker import canonical_hole, _board_texture


def _street_label(board_size: int) -> str:
    if board_size == 0:
        return "PF"
    if board_size == 3:
        return "F"
    if board_size == 4:
        return "T"
    return "R"


def _pot_bucket(pot: int, big_blind: int) -> str:
    ratio = pot / big_blind if big_blind > 0 else 0
    if ratio <= 3:
        return "tiny"
    if ratio <= 8:
        return "small"
    if ratio <= 20:
        return "medium"
    if ratio <= 50:
        return "large"
    return "huge"


def _spr_bucket(stack: int, pot: int) -> str:
    if pot <= 0:
        return "deep"
    spr = stack / pot
    if spr < 2:
        return "commit"
    if spr < 5:
        return "shallow"
    if spr < 12:
        return "medium"
    return "deep"


def _action_sequence(state: HandState, current_street_only: bool = True) -> str:
    """Compact encoding of the action history."""
    from packages.common.types import ActionType

    abbrev = {
        ActionType.FOLD: "f",
        ActionType.CHECK: "x",
        ActionType.CALL: "c",
        ActionType.BET: "b",
        ActionType.RAISE: "r",
        ActionType.ALL_IN: "a",
        ActionType.POST_SMALL_BLIND: "",
        ActionType.POST_BIG_BLIND: "",
    }

    parts: list[str] = []
    for ev in state.actions:
        s = abbrev.get(ev.action_type, "?")
        if s:
            parts.append(s)

    seq = "".join(parts)
    return seq[-8:] if len(seq) > 8 else seq


def build_info_set_key(state: HandState, seat: int) -> str:
    """Build a compact information set key for the acting player.

    Format: ``hole|street|position|spr|pot_bucket|texture|actions``

    This abstracts the game state into ~5000-20000 unique info sets,
    which is tractable for tabular CFR.
    """
    player = state.players[seat]
    hole = canonical_hole(player.hole_cards)
    street = _street_label(len(state.board))
    position = "IP" if seat == state.button_seat else "OOP"
    pot_b = _pot_bucket(state.pot, state.big_blind)
    spr_b = _spr_bucket(player.stack, state.pot)
    texture = _board_texture(state.board)
    actions = _action_sequence(state)

    return f"{hole}|{street}|{position}|{spr_b}|{pot_b}|{texture}|{actions}"
