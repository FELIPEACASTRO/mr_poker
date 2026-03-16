from __future__ import annotations

from dataclasses import dataclass

from packages.engine.models import Card
from packages.features.poker import RANK_ORDER


@dataclass(frozen=True)
class BoardTexture:
    """Detailed board texture analysis."""

    wetness: float
    is_monotone: bool
    is_two_tone: bool
    is_rainbow: bool
    is_paired: bool
    pair_count: int
    flush_possible: bool
    flush_draw_possible: bool
    straight_possible: bool
    straight_draw_possible: bool
    high_card_rank: int
    connectivity: float


def analyze_board(board: list[Card]) -> BoardTexture:
    """Analyze board texture for strategic decision making."""
    if len(board) < 3:
        return BoardTexture(
            wetness=0.0,
            is_monotone=False,
            is_two_tone=False,
            is_rainbow=False,
            is_paired=False,
            pair_count=0,
            flush_possible=False,
            flush_draw_possible=False,
            straight_possible=False,
            straight_draw_possible=False,
            high_card_rank=0,
            connectivity=0.0,
        )

    suits = [c.suit for c in board]
    ranks = sorted([RANK_ORDER[c.rank] for c in board])
    unique_suits = len(set(suits))
    suit_counts = {s: suits.count(s) for s in set(suits)}
    max_suit_count = max(suit_counts.values())

    # Pairing
    unique_ranks = len(set(ranks))
    pair_count = len(ranks) - unique_ranks
    is_paired = pair_count > 0

    # Suit texture
    is_monotone = unique_suits == 1
    is_two_tone = unique_suits == 2
    is_rainbow = unique_suits >= 3

    # Flush analysis
    flush_possible = max_suit_count >= 3 and len(board) >= 3
    flush_draw_possible = max_suit_count >= 2

    # Straight analysis
    unique_sorted = sorted(set(ranks))
    straight_possible = False
    straight_draw_possible = False
    if len(unique_sorted) >= 3:
        for i in range(len(unique_sorted) - 2):
            window = unique_sorted[i : i + min(5, len(unique_sorted) - i)]
            span = window[-1] - window[0]
            if len(window) >= 5 and span == 4:
                straight_possible = True
            if span <= 4:
                straight_draw_possible = True

    # Connectivity: average gap between consecutive sorted ranks
    gaps = [unique_sorted[i + 1] - unique_sorted[i] for i in range(len(unique_sorted) - 1)]
    connectivity = 1.0 / (1.0 + sum(gaps) / len(gaps)) if gaps else 0.0

    # Wetness: composite score (0-1) indicating how draw-heavy the board is
    wetness = 0.0
    if flush_draw_possible:
        wetness += 0.3
    if flush_possible:
        wetness += 0.2
    if straight_draw_possible:
        wetness += 0.2
    if straight_possible:
        wetness += 0.1
    if connectivity > 0.5:
        wetness += 0.1
    if not is_paired:
        wetness += 0.1

    return BoardTexture(
        wetness=min(1.0, wetness),
        is_monotone=is_monotone,
        is_two_tone=is_two_tone,
        is_rainbow=is_rainbow,
        is_paired=is_paired,
        pair_count=pair_count,
        flush_possible=flush_possible,
        flush_draw_possible=flush_draw_possible,
        straight_possible=straight_possible,
        straight_draw_possible=straight_draw_possible,
        high_card_rank=max(ranks) if ranks else 0,
        connectivity=round(connectivity, 4),
    )
