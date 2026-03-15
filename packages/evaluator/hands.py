from __future__ import annotations

from collections import Counter
from functools import lru_cache
from itertools import combinations
from typing import Any

from packages.engine.models import Card

try:
    import eval7  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional perf dependency
    eval7 = None
try:
    from treys import Card as TreysCard  # type: ignore[import-untyped]
    from treys import Evaluator as TreysEvaluator
except Exception:  # pragma: no cover - optional perf dependency
    TreysCard = None
    TreysEvaluator = None

CATEGORY_LABELS = {
    8: "straight_flush",
    7: "four_of_a_kind",
    6: "full_house",
    5: "flush",
    4: "straight",
    3: "three_of_a_kind",
    2: "two_pair",
    1: "one_pair",
    0: "high_card",
}

RANK_TO_VALUE = {r: i for i, r in enumerate("..23456789TJQKA")}
HANDTYPE_TO_CATEGORY = {
    "high_card": 0,
    "pair": 1,
    "one_pair": 1,
    "two_pair": 2,
    "trips": 3,
    "three_of_a_kind": 3,
    "straight": 4,
    "flush": 5,
    "full_house": 6,
    "quads": 7,
    "four_of_a_kind": 7,
    "straight_flush": 8,
}
TREYS_CLASS_TO_CATEGORY = {
    0: 8,  # Royal Flush (treys-specific subclass of straight flush)
    1: 8,  # Straight Flush
    2: 7,  # Four of a Kind
    3: 6,  # Full House
    4: 5,  # Flush
    5: 4,  # Straight
    6: 3,  # Three of a Kind
    7: 2,  # Two Pair
    8: 1,  # One Pair
    9: 0,  # High Card
}
TREYS_EVALUATOR = TreysEvaluator() if TreysEvaluator is not None else None


def _straight_high(values: list[int]) -> int | None:
    uniq = sorted(set(values), reverse=True)
    if 14 in uniq:
        uniq.append(1)
    for idx in range(len(uniq) - 4):
        window = uniq[idx : idx + 5]
        if window[0] - window[4] == 4 and len(set(window)) == 5:
            return window[0]
    return None


@lru_cache(maxsize=52)
def _eval7_card_from_str(card_repr: str) -> Any:
    if eval7 is None:
        raise RuntimeError("eval7 unavailable")
    return eval7.Card(card_repr)


def _normalize_handtype(handtype: str) -> str:
    return handtype.strip().lower().replace(" ", "_")


@lru_cache(maxsize=52)
def _treys_card_from_str(card_repr: str) -> int:
    if TreysCard is None:
        raise RuntimeError("treys unavailable")
    return int(TreysCard.new(card_repr))


def _rank_with_eval7(cards: list[Card]) -> tuple[int, tuple[int, ...]]:
    if eval7 is None:
        raise RuntimeError("eval7 unavailable")
    score = int(eval7.evaluate([_eval7_card_from_str(str(card)) for card in cards]))
    handtype = _normalize_handtype(str(eval7.handtype(score)))
    category = HANDTYPE_TO_CATEGORY.get(handtype)
    if category is None:
        raise ValueError(f"unsupported eval7 handtype: {handtype}")
    return category, (score,)


def _rank_with_treys(cards: list[Card]) -> tuple[int, tuple[int, ...]]:
    if TREYS_EVALUATOR is None:
        raise RuntimeError("treys unavailable")
    ints = [_treys_card_from_str(str(card)) for card in cards]
    hand = ints[:2]
    board = ints[2:]
    score = int(TREYS_EVALUATOR.evaluate(board, hand))
    rank_class = int(TREYS_EVALUATOR.get_rank_class(score))
    category = TREYS_CLASS_TO_CATEGORY.get(rank_class)
    if category is None:
        raise ValueError(f"unsupported treys rank class: {rank_class}")
    # treys uses lower-is-better rank values; invert to preserve higher-is-better tuple compare.
    return category, (-score,)


def evaluate_five(cards: list[Card]) -> tuple[int, tuple[int, ...]]:
    if len(cards) != 5:
        raise ValueError("evaluate_five requires exactly 5 cards")
    if eval7 is not None:
        return _rank_with_eval7(cards)
    if TREYS_EVALUATOR is not None:
        return _rank_with_treys(cards)

    values = sorted((RANK_TO_VALUE[c.rank] for c in cards), reverse=True)
    suits = [c.suit for c in cards]
    counts = Counter(values)
    ordered = sorted(counts.items(), key=lambda item: (item[1], item[0]), reverse=True)
    flush = len(set(suits)) == 1
    straight_high = _straight_high(values)

    if flush and straight_high is not None:
        return 8, (straight_high,)
    if ordered[0][1] == 4:
        four = ordered[0][0]
        kicker = max(v for v in values if v != four)
        return 7, (four, kicker)
    if ordered[0][1] == 3 and ordered[1][1] == 2:
        return 6, (ordered[0][0], ordered[1][0])
    if flush:
        return 5, tuple(sorted(values, reverse=True))
    if straight_high is not None:
        return 4, (straight_high,)
    if ordered[0][1] == 3:
        trips = ordered[0][0]
        kickers = tuple(sorted((v for v in values if v != trips), reverse=True))
        return 3, (trips, *kickers)
    if ordered[0][1] == 2 and ordered[1][1] == 2:
        pair_high = max(ordered[0][0], ordered[1][0])
        pair_low = min(ordered[0][0], ordered[1][0])
        kicker = max(v for v in values if v not in {pair_high, pair_low})
        return 2, (pair_high, pair_low, kicker)
    if ordered[0][1] == 2:
        pair = ordered[0][0]
        kickers = tuple(sorted((v for v in values if v != pair), reverse=True))
        return 1, (pair, *kickers)
    return 0, tuple(sorted(values, reverse=True))


def best_hand_rank(cards: list[Card]) -> tuple[int, tuple[int, ...]]:
    if len(cards) < 5:
        raise ValueError("need at least 5 cards")
    if eval7 is not None:
        # eval7 natively handles 5-7 cards and picks the best 5-card hand.
        return _rank_with_eval7(cards)
    if TREYS_EVALUATOR is not None and len(cards) <= 7:
        return _rank_with_treys(cards)
    return max(evaluate_five(list(combo)) for combo in combinations(cards, 5))


def hand_label(rank: tuple[int, tuple[int, ...]]) -> str:
    return CATEGORY_LABELS[rank[0]]
