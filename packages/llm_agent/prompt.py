"""Build PokerBench-format prompts from mr_poker game state.

Reconstructs the natural-language instruction format used by the PokerBench
dataset so that an LLM fine-tuned on PokerBench will produce optimal actions.
"""

from __future__ import annotations

from typing import Any

_RANK_NAMES: dict[str, str] = {
    "A": "Ace", "K": "King", "Q": "Queen", "J": "Jack", "T": "Ten",
    "9": "Nine", "8": "Eight", "7": "Seven", "6": "Six", "5": "Five",
    "4": "Four", "3": "Three", "2": "Two",
}

_SUIT_NAMES: dict[str, str] = {
    "h": "Heart", "d": "Diamond", "c": "Club", "s": "Spade",
}


def _card_to_text(card_str: str) -> str:
    """Convert 'Kh' -> 'King of Heart'."""
    if len(card_str) != 2:
        return card_str
    rank = _RANK_NAMES.get(card_str[0], card_str[0])
    suit = _SUIT_NAMES.get(card_str[1], card_str[1])
    return f"{rank} of {suit}"


def build_pokerbench_prompt(
    *,
    position: str,
    hole_cards: list[str],
    board: list[str],
    pot: float,
    street: str,
    actions_text: str = "",
) -> str:
    """Build a PokerBench-style instruction prompt.

    Parameters match the fields available from a normalized game state.
    """
    card1 = _card_to_text(hole_cards[0]) if len(hole_cards) > 0 else "Unknown"
    card2 = _card_to_text(hole_cards[1]) if len(hole_cards) > 1 else "Unknown"
    holding = f"[{card1} and {card2}]"

    parts = [
        "You are a specialist in playing 6-handed No Limit Texas Holdem. "
        "The following will be a game scenario and you need to make the optimal decision. "
        "Here is a game summary: The small blind is 0.5 chips and the big blind is 1 chips. "
        "Everyone started with 100 chips. The player positions involved in this game are "
        "UTG, HJ, CO, BTN, SB, BB. "
        f"In this hand, your position is {position}, and your holding is {holding}.",
    ]

    if actions_text:
        parts.append(actions_text)

    # Board cards
    if len(board) >= 3:
        flop = [_card_to_text(c) for c in board[:3]]
        parts.append(
            f"The flop comes {flop[0]}, {flop[1]}, and {flop[2]}."
        )
    if len(board) >= 4:
        parts.append(f"The turn comes {_card_to_text(board[3])}.")
    if len(board) >= 5:
        parts.append(f"The river comes {_card_to_text(board[4])}.")

    parts.append(
        f"Now it is your turn to make a move. "
        f"To remind you, the current pot size is {pot} chips, "
        f"and your holding is {holding}. "
        f"Decide on an action based on the strength of your hand on this board, "
        f"your position, and actions before you. "
        f"Do not explain your answer. Your optimal action is:"
    )

    return " ".join(parts)


def build_prompt_from_spot(spot: dict[str, Any]) -> str:
    """Build a prompt from a normalized spot dictionary."""
    position = "BTN" if spot.get("actor_seat") == spot.get("button_seat", 0) else "BB"
    hole_cards = spot.get("hole_cards", [])
    board = spot.get("board", [])
    pot = spot.get("pot", 0.0)
    street = spot.get("street", "pre_flop")

    return build_pokerbench_prompt(
        position=position,
        hole_cards=hole_cards,
        board=board,
        pot=pot,
        street=street,
    )
