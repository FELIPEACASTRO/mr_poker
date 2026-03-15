from packages.engine.deck import Deck


def test_deck_starts_with_52_unique_cards() -> None:
    deck = Deck(seed=42)
    cards = deck.cards
    assert len(cards) == 52
    assert len({str(c) for c in cards}) == 52


def test_draw_reduces_size() -> None:
    deck = Deck(seed=42)
    deck.shuffle()
    drawn = deck.draw(2)
    assert len(drawn) == 2
    assert len(deck.cards) == 50
