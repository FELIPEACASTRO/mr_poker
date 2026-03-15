from __future__ import annotations

import pytest

from packages.engine.models import Card
from packages.evaluator import hands


def _cards(*values: str) -> list[Card]:
    return [Card.from_str(value) for value in values]


def _clear_hands_caches() -> None:
    hands._eval7_card_from_str.cache_clear()
    hands._treys_card_from_str.cache_clear()


class _FakeEval7:
    @staticmethod
    def Card(card_repr: str) -> str:
        return card_repr

    @staticmethod
    def evaluate(cards: list[str]) -> int:
        return sum((ord(card[0]) * 7) + ord(card[1]) for card in cards)

    @staticmethod
    def handtype(score: int) -> str:  # noqa: ARG004 - signature mirrors eval7
        return "Full House"


class _UnsupportedHandtypeEval7(_FakeEval7):
    @staticmethod
    def handtype(score: int) -> str:  # noqa: ARG004 - signature mirrors eval7
        return "mystery_type"


class _FakeTreysCard:
    @staticmethod
    def new(card_repr: str) -> int:
        return int(sum(ord(ch) for ch in card_repr))


class _FakeTreysEvaluator:
    def __init__(self, rank_class: int = 7) -> None:
        self.rank_class = rank_class

    def evaluate(self, board: list[int], hand: list[int]) -> int:
        return int(sum(board) + sum(hand))

    def get_rank_class(self, score: int) -> int:  # noqa: ARG002 - API compatibility
        return self.rank_class


def test_helper_functions_and_error_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hands, "eval7", None)
    monkeypatch.setattr(hands, "TreysCard", None)
    _clear_hands_caches()

    assert hands._straight_high([14, 5, 4, 3, 2]) == 5
    assert hands._straight_high([14, 14, 8, 5, 3]) is None
    assert hands._normalize_handtype(" Full House ") == "full_house"

    with pytest.raises(RuntimeError, match="eval7 unavailable"):
        hands._eval7_card_from_str("As")

    with pytest.raises(RuntimeError, match="treys unavailable"):
        hands._treys_card_from_str("As")

    with pytest.raises(ValueError, match="exactly 5 cards"):
        hands.evaluate_five(_cards("As", "Ks", "Qs", "Js"))

    with pytest.raises(ValueError, match="need at least 5 cards"):
        hands.best_hand_rank(_cards("As", "Ah", "Kd", "Qc"))


def test_eval7_and_treys_ranking_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hands, "eval7", _FakeEval7)
    _clear_hands_caches()

    eval7_rank = hands._rank_with_eval7(_cards("As", "Ah", "Kd", "Qc", "2d"))
    assert eval7_rank[0] == 6
    assert isinstance(eval7_rank[1][0], int)

    best_eval7 = hands.best_hand_rank(_cards("As", "Ah", "Kd", "Qc", "2d", "3h", "4s"))
    assert best_eval7[0] == 6

    monkeypatch.setattr(hands, "eval7", _UnsupportedHandtypeEval7)
    _clear_hands_caches()
    with pytest.raises(ValueError, match="unsupported eval7 handtype"):
        hands._rank_with_eval7(_cards("As", "Ah", "Kd", "Qc", "2d"))

    monkeypatch.setattr(hands, "eval7", None)
    monkeypatch.setattr(hands, "TreysCard", _FakeTreysCard)
    monkeypatch.setattr(hands, "TREYS_EVALUATOR", _FakeTreysEvaluator(rank_class=7))
    _clear_hands_caches()

    treys_rank = hands._rank_with_treys(_cards("As", "Ah", "Kd", "Qc", "2d"))
    assert treys_rank[0] == 2

    best_treys = hands.best_hand_rank(_cards("As", "Ah", "Kd", "Qc", "2d", "3h"))
    assert best_treys[0] == 2

    monkeypatch.setattr(hands, "TREYS_EVALUATOR", None)
    with pytest.raises(RuntimeError, match="treys unavailable"):
        hands._rank_with_treys(_cards("As", "Ah", "Kd", "Qc", "2d"))

    monkeypatch.setattr(hands, "TREYS_EVALUATOR", _FakeTreysEvaluator(rank_class=99))
    with pytest.raises(ValueError, match="unsupported treys rank class"):
        hands._rank_with_treys(_cards("As", "Ah", "Kd", "Qc", "2d"))


@pytest.mark.parametrize(
    ("cards", "expected_category"),
    [
        (("As", "Ks", "Qs", "Js", "Ts"), 8),  # straight flush
        (("Ah", "Ad", "Ac", "As", "2d"), 7),  # quads
        (("Ah", "Ad", "Ac", "Kh", "Kd"), 6),  # full house
        (("Ah", "Jh", "8h", "3h", "2h"), 5),  # flush
        (("9h", "8d", "7s", "6c", "5h"), 4),  # straight
        (("Ah", "Ad", "Ac", "Kh", "Qd"), 3),  # trips
        (("Ah", "Ad", "Kh", "Kd", "2c"), 2),  # two pair
        (("Ah", "Ad", "Kc", "Qd", "2s"), 1),  # one pair
        (("Ah", "Kd", "Qc", "9s", "2h"), 0),  # high card
    ],
)
def test_evaluate_five_python_fallback_all_categories(
    monkeypatch: pytest.MonkeyPatch,
    cards: tuple[str, ...],
    expected_category: int,
) -> None:
    monkeypatch.setattr(hands, "eval7", None)
    monkeypatch.setattr(hands, "TREYS_EVALUATOR", None)
    _clear_hands_caches()

    rank = hands.evaluate_five(_cards(*cards))
    assert rank[0] == expected_category
    assert hands.hand_label(rank) == hands.CATEGORY_LABELS[expected_category]


def test_best_hand_rank_python_combinatorics_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(hands, "eval7", None)
    monkeypatch.setattr(hands, "TREYS_EVALUATOR", None)
    _clear_hands_caches()

    # Best 5-card hand here is full house: AAAKK
    rank = hands.best_hand_rank(_cards("Ah", "Ad", "Ac", "Kh", "Kd", "2s", "3c"))
    assert rank[0] == 6
    assert hands.hand_label(rank) == "full_house"
