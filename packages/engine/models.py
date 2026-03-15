from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from packages.common.types import ActionType, Street

RANKS = "23456789TJQKA"
SUITS = "cdhs"


@dataclass(frozen=True)
class Card:
    rank: str
    suit: str

    def __post_init__(self) -> None:
        if self.rank not in RANKS:
            raise ValueError(f"invalid rank: {self.rank}")
        if self.suit not in SUITS:
            raise ValueError(f"invalid suit: {self.suit}")

    def __str__(self) -> str:
        return f"{self.rank}{self.suit}"

    @classmethod
    def from_str(cls, value: str) -> "Card":
        if len(value) != 2:
            raise ValueError("card string must have exactly 2 characters")
        return cls(rank=value[0], suit=value[1])


@dataclass
class PlayerState:
    seat: int
    stack: int
    invested_this_round: int = 0
    total_invested: int = 0
    folded: bool = False
    is_all_in: bool = False
    hole_cards: list[Card] = field(default_factory=list)


@dataclass
class ActionEvent:
    actor_seat: int
    action_type: ActionType
    amount: int = 0
    street: Street = Street.PRE_FLOP
    note: str = ""


@dataclass
class HandState:
    hand_id: str
    button_seat: int
    street: Street
    pot: int
    to_call: int
    min_raise_to: Optional[int]
    board: list[Card] = field(default_factory=list)
    actions: list[ActionEvent] = field(default_factory=list)
    players: dict[int, PlayerState] = field(default_factory=dict)
    small_blind: int = 1
    big_blind: int = 2
    current_bet: int = 0
    acting_seat: Optional[int] = None
    street_starting_seat: Optional[int] = None
    last_aggressor_seat: Optional[int] = None
    acted_this_street: set[int] = field(default_factory=set)
    winner_seat: Optional[int] = None
    is_terminal: bool = False
    showdown_reached: bool = False
