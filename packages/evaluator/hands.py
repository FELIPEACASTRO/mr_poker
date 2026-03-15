from __future__ import annotations

from collections import Counter
from itertools import combinations

from packages.engine.models import Card

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


def _straight_high(values: list[int]) -> int | None:
    uniq = sorted(set(values), reverse=True)
    if 14 in uniq:
        uniq.append(1)
    for idx in range(len(uniq) - 4):
        window = uniq[idx : idx + 5]
        if window[0] - window[4] == 4 and len(set(window)) == 5:
            return window[0]
    return None


def evaluate_five(cards: list[Card]) -> tuple[int, tuple[int, ...]]:
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
    return max(evaluate_five(list(combo)) for combo in combinations(cards, 5))


def hand_label(rank: tuple[int, tuple[int, ...]]) -> str:
    return CATEGORY_LABELS[rank[0]]
