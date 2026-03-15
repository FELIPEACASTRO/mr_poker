from __future__ import annotations

import random
from functools import lru_cache
from itertools import combinations
from typing import Any

from packages.engine.models import Card
from packages.evaluator.hands import best_hand_rank

try:
    import eval7  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional perf dependency
    eval7 = None

RANKS = "23456789TJQKA"
SUITS = "cdhs"
FULL_DECK_STR = tuple(f"{rank}{suit}" for rank in RANKS for suit in SUITS)


def _canonical_cards(cards: list[Card]) -> tuple[str, ...]:
    return tuple(sorted(str(card) for card in cards))


@lru_cache(maxsize=52)
def _eval7_card(card_repr: str) -> Any:
    if eval7 is None:
        raise RuntimeError("eval7 unavailable")
    return eval7.Card(card_repr)


def _score_python(cards: list[Card]) -> tuple[int, tuple[int, ...]]:
    return best_hand_rank(cards)


def _score_eval7(cards: tuple[Any, ...]) -> int:
    if eval7 is None:
        raise RuntimeError("eval7 unavailable")
    return int(eval7.evaluate(list(cards)))


def _estimate_equity_eval7(
    *,
    hero_hole: tuple[str, ...],
    board: tuple[str, ...],
    samples: int,
    seed: int | None,
) -> float:
    hero_hole_eval = tuple(_eval7_card(card) for card in hero_hole)
    board_eval = tuple(_eval7_card(card) for card in board)
    dead = set(hero_hole) | set(board)
    remaining = tuple(_eval7_card(card) for card in FULL_DECK_STR if card not in dead)

    wins = 0
    ties = 0
    total = 0

    if len(board_eval) == 5:
        hero_score = _score_eval7(hero_hole_eval + board_eval)
        for opp_hole in combinations(remaining, 2):
            opp_score = _score_eval7(tuple(opp_hole) + board_eval)
            total += 1
            if hero_score > opp_score:
                wins += 1
            elif hero_score == opp_score:
                ties += 1
        return (wins + 0.5 * ties) / total if total else 0.0

    # Exact turn evaluation (heads-up): enumerate river and opponent combinations.
    if len(board_eval) == 4:
        for river_index, river in enumerate(remaining):
            board_full = board_eval + (river,)
            hero_score = _score_eval7(hero_hole_eval + board_full)
            others = remaining[:river_index] + remaining[river_index + 1 :]
            for opp_hole in combinations(others, 2):
                opp_score = _score_eval7(tuple(opp_hole) + board_full)
                total += 1
                if hero_score > opp_score:
                    wins += 1
                elif hero_score == opp_score:
                    ties += 1
        return (wins + 0.5 * ties) / total if total else 0.0

    rng = random.Random(seed)
    need_board = 5 - len(board_eval)
    for _ in range(samples):
        draw = rng.sample(remaining, 2 + need_board)
        opp_hole = tuple(draw[:2])
        runout = tuple(draw[2:])
        board_full = board_eval + runout
        hero_score = _score_eval7(hero_hole_eval + board_full)
        opp_score = _score_eval7(opp_hole + board_full)
        total += 1
        if hero_score > opp_score:
            wins += 1
        elif hero_score == opp_score:
            ties += 1
    return (wins + 0.5 * ties) / total if total else 0.0


def _estimate_equity_python(
    *,
    hero_hole: tuple[str, ...],
    board: tuple[str, ...],
    samples: int,
    seed: int | None,
) -> float:
    hero_cards = [Card.from_str(card) for card in hero_hole]
    board_cards = [Card.from_str(card) for card in board]
    dead = set(hero_hole) | set(board)
    remaining = [Card.from_str(card) for card in FULL_DECK_STR if card not in dead]
    rng = random.Random(seed)

    wins = 0
    ties = 0
    total = 0

    if len(board_cards) == 5:
        hero_rank = _score_python(hero_cards + board_cards)
        for opp_hole in combinations(remaining, 2):
            opp_rank = _score_python(list(opp_hole) + board_cards)
            total += 1
            if hero_rank > opp_rank:
                wins += 1
            elif hero_rank == opp_rank:
                ties += 1
        return (wins + 0.5 * ties) / total if total else 0.0

    need_board = 5 - len(board_cards)
    for _ in range(samples):
        draw = rng.sample(remaining, 2 + need_board)
        opp_hole_cards = draw[:2]
        runout = draw[2:]
        board_full = board_cards + runout
        hero_rank = _score_python(hero_cards + board_full)
        opp_rank = _score_python(opp_hole_cards + board_full)
        total += 1
        if hero_rank > opp_rank:
            wins += 1
        elif hero_rank == opp_rank:
            ties += 1
    return (wins + 0.5 * ties) / total if total else 0.0


@lru_cache(maxsize=50_000)
def _estimate_equity_cached(
    hero_hole: tuple[str, ...],
    board: tuple[str, ...],
    num_opponents: int,
    samples: int,
    seed: int | None,
) -> float:
    if num_opponents != 1:
        raise ValueError("Sprint 04 estimator currently supports exactly 1 opponent")
    if len(hero_hole) != 2:
        raise ValueError("hero_hole must contain exactly 2 cards")
    if len(board) > 5:
        raise ValueError("board cannot exceed 5 cards")
    if eval7 is not None:
        return _estimate_equity_eval7(
            hero_hole=hero_hole, board=board, samples=samples, seed=seed
        )
    return _estimate_equity_python(
        hero_hole=hero_hole, board=board, samples=samples, seed=seed
    )


def estimate_equity(
    hero_hole: list[Card],
    board: list[Card],
    *,
    num_opponents: int = 1,
    samples: int = 200,
    seed: int | None = None,
) -> float:
    """Estimate showdown equity for heads-up settings.

    Uses eval7 when available for high-performance hand evaluation, with a
    deterministic fallback path to the local evaluator.
    """
    hero_key = _canonical_cards(hero_hole)
    board_key = _canonical_cards(board)
    return _estimate_equity_cached(hero_key, board_key, num_opponents, samples, seed)
