
from __future__ import annotations

import random
from itertools import combinations

from packages.engine.models import Card
from packages.evaluator.hands import best_hand_rank

RANKS = '23456789TJQKA'
SUITS = 'cdhs'


def _full_deck() -> list[Card]:
    return [Card(rank=r, suit=s) for r in RANKS for s in SUITS]


def estimate_equity(
    hero_hole: list[Card],
    board: list[Card],
    *,
    num_opponents: int = 1,
    samples: int = 200,
    seed: int | None = None,
) -> float:
    """Estimate showdown equity for heads-up / small-opponent-count settings.

    This is intentionally lightweight and deterministic for local experimentation.
    It is not the final solver-grade equity engine.
    """
    if num_opponents != 1:
        raise ValueError('Sprint 04 estimator currently supports exactly 1 opponent')
    if len(hero_hole) != 2:
        raise ValueError('hero_hole must contain exactly 2 cards')
    if len(board) > 5:
        raise ValueError('board cannot exceed 5 cards')

    dead = {str(c) for c in hero_hole + board}
    remaining = [c for c in _full_deck() if str(c) not in dead]
    rng = random.Random(seed)

    # Exact enumeration when possible on river/turn with small state space.
    if len(board) == 5:
        wins = ties = total = 0
        hero_rank = best_hand_rank(hero_hole + board)
        for opp_hole in combinations(remaining, 2):
            opp_rank = best_hand_rank(list(opp_hole) + board)
            total += 1
            if hero_rank > opp_rank:
                wins += 1
            elif hero_rank == opp_rank:
                ties += 1
        return (wins + 0.5 * ties) / total if total else 0.0

    wins = ties = 0
    total = 0
    need_board = 5 - len(board)
    for _ in range(samples):
        sample = remaining[:]
        rng.shuffle(sample)
        opp_hole = sample[:2]
        runout = sample[2:2 + need_board]
        full_board = board + runout
        hero_rank = best_hand_rank(hero_hole + full_board)
        opp_rank = best_hand_rank(list(opp_hole) + full_board)
        total += 1
        if hero_rank > opp_rank:
            wins += 1
        elif hero_rank == opp_rank:
            ties += 1
    return (wins + 0.5 * ties) / total if total else 0.0
