from __future__ import annotations

import random

from packages.engine.models import Card
from packages.equity.monte_carlo import (
    FULL_DECK_STR,
    estimate_equity,
)


# Common preflop hand ranges
RANGES: dict[str, list[str]] = {
    "premium": ["AA", "KK", "QQ", "JJ", "AKs", "AKo"],
    "strong": [
        "AA", "KK", "QQ", "JJ", "TT", "AKs", "AKo", "AQs", "AQo",
        "AJs", "KQs",
    ],
    "open_raise": [
        "AA", "KK", "QQ", "JJ", "TT", "99", "88",
        "AKs", "AKo", "AQs", "AQo", "AJs", "AJo", "ATs",
        "KQs", "KQo", "KJs", "KTs", "QJs", "QTs", "JTs",
    ],
    "wide": [
        "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66", "55",
        "AKs", "AKo", "AQs", "AQo", "AJs", "AJo", "ATs", "ATo",
        "A9s", "A8s", "A7s", "A6s", "A5s", "A4s", "A3s", "A2s",
        "KQs", "KQo", "KJs", "KJo", "KTs", "K9s",
        "QJs", "QJo", "QTs", "Q9s",
        "JTs", "JTo", "J9s",
        "T9s", "T8s", "98s", "87s", "76s", "65s", "54s",
    ],
}


def range_vs_range_equity(
    range_a: list[str],
    range_b: list[str],
    board: list[Card] | None = None,
    *,
    samples: int = 100,
    seed: int = 42,
) -> dict[str, float]:
    """Estimate equity of range A vs range B.

    Returns dict with keys: range_a_equity, range_b_equity, samples_used.
    """
    board = board or []
    rng = random.Random(seed)
    board_strs = {str(c) for c in board}
    deck = [c for c in FULL_DECK_STR if c not in board_strs]

    a_wins = 0
    b_wins = 0
    ties = 0
    total = 0

    for _ in range(samples):
        # Sample a hand from each range
        hand_a = _sample_hand_from_range(range_a, deck, rng)
        if hand_a is None:
            continue
        remaining = [c for c in deck if c not in hand_a]
        hand_b = _sample_hand_from_range(range_b, remaining, rng)
        if hand_b is None:
            continue

        cards_a = [Card.from_str(c) for c in hand_a]
        [Card.from_str(c) for c in hand_b]

        eq_a = estimate_equity(cards_a, board, samples=20, seed=rng.randint(0, 999999))
        eq_b = 1.0 - eq_a  # Approximation for heads-up

        total += 1
        if eq_a > 0.5:
            a_wins += 1
        elif eq_b > 0.5:
            b_wins += 1
        else:
            ties += 1

    if total == 0:
        return {"range_a_equity": 0.5, "range_b_equity": 0.5, "samples_used": 0}

    return {
        "range_a_equity": round((a_wins + 0.5 * ties) / total, 4),
        "range_b_equity": round((b_wins + 0.5 * ties) / total, 4),
        "samples_used": total,
    }


def _sample_hand_from_range(
    hand_range: list[str],
    deck: list[str],
    rng: random.Random,
) -> tuple[str, str] | None:
    """Sample a specific hand from a range notation."""
    rng.shuffle(hand_range)
    {r: i for i, r in enumerate("23456789TJQKA", start=2)}

    for notation in hand_range:
        if len(notation) == 2:
            # Pair like "AA"
            rank = notation[0]
            candidates = [c for c in deck if c[0] == rank]
            if len(candidates) >= 2:
                pair = rng.sample(candidates, 2)
                return (pair[0], pair[1])
        elif len(notation) == 3:
            r1, r2, suitedness = notation[0], notation[1], notation[2]
            if suitedness == "s":
                for suit in "cdhs":
                    c1, c2 = f"{r1}{suit}", f"{r2}{suit}"
                    if c1 in deck and c2 in deck:
                        return (c1, c2)
            else:
                c1_list = [c for c in deck if c[0] == r1]
                c2_list = [c for c in deck if c[0] == r2]
                for c1 in c1_list:
                    for c2 in c2_list:
                        if c1[1] != c2[1] and c2 in deck:
                            return (c1, c2)
    return None
