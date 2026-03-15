from __future__ import annotations

import random
from typing import Iterable

from packages.engine.models import Card, RANKS, SUITS


class Deck:
    def __init__(self, seed: int | None = None, top_cards: list[Card] | None = None) -> None:
        self._rng = random.Random(seed)
        all_cards = [Card(rank=r, suit=s) for r in RANKS for s in SUITS]
        top_cards = top_cards or []
        seen = {str(card) for card in top_cards}
        if len(seen) != len(top_cards):
            raise ValueError("duplicate cards in forced deck prefix")
        remaining = [card for card in all_cards if str(card) not in seen]
        self._rng.shuffle(remaining)
        self._cards = [*top_cards, *remaining]

    @property
    def cards(self) -> list[Card]:
        return list(self._cards)

    def shuffle(self) -> None:
        self._rng.shuffle(self._cards)

    def draw(self, count: int = 1) -> list[Card]:
        if count < 0:
            raise ValueError("count must be non-negative")
        if count > len(self._cards):
            raise ValueError("not enough cards remaining")
        drawn = self._cards[:count]
        self._cards = self._cards[count:]
        return drawn

    def remove(self, cards: Iterable[Card]) -> None:
        for card in cards:
            self._cards.remove(card)

    @classmethod
    def from_card_strings(cls, card_strings: list[str], seed: int | None = None) -> "Deck":
        return cls(seed=seed, top_cards=[Card.from_str(value) for value in card_strings])
