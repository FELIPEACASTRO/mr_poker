
from __future__ import annotations

from dataclasses import dataclass

from packages.engine.models import Card, HandState, PlayerState

RANK_ORDER = {r: i for i, r in enumerate('23456789TJQKA', start=2)}


@dataclass
class BaselineFeatures:
    street: str
    to_call: int
    pot: int
    stack: int
    hole_class: str
    is_pair: bool
    is_suited: bool
    is_connected: bool
    high_rank: int
    low_rank: int
    board_size: int
    pot_odds: float
    board_texture: str
    spr: float


def canonical_hole(cards: list[Card]) -> str:
    if len(cards) != 2:
        return '??'
    c1, c2 = sorted(cards, key=lambda c: RANK_ORDER[c.rank], reverse=True)
    suited = 's' if c1.suit == c2.suit else 'o'
    if c1.rank == c2.rank:
        return f'{c1.rank}{c2.rank}'
    return f'{c1.rank}{c2.rank}{suited}'


def _board_texture(board: list[Card]) -> str:
    if len(board) < 3:
        return 'preflop'
    suits = [c.suit for c in board]
    ranks = sorted(RANK_ORDER[c.rank] for c in board)
    unique_suits = len(set(suits))
    paired = len(set(ranks)) < len(ranks)
    span = max(ranks) - min(ranks)
    connectivity = 'connected' if span <= 4 else 'disconnected'
    suit_label = 'monotone' if unique_suits == 1 else 'two-tone' if unique_suits == 2 else 'rainbow'
    paired_label = 'paired' if paired else 'unpaired'
    return f'{suit_label}-{paired_label}-{connectivity}'


def derive_baseline_features(state: HandState, player: PlayerState) -> BaselineFeatures:
    ordered = sorted(player.hole_cards, key=lambda c: RANK_ORDER[c.rank], reverse=True)
    high = RANK_ORDER[ordered[0].rank]
    low = RANK_ORDER[ordered[1].rank]
    to_call = max(0, state.current_bet - player.invested_this_round)
    pot_odds = (to_call / (state.pot + to_call)) if to_call > 0 else 0.0
    effective_stack = min(p.stack for p in state.players.values() if not p.folded)
    spr = (effective_stack / state.pot) if state.pot > 0 else 999.0
    return BaselineFeatures(
        street=state.street.value,
        to_call=to_call,
        pot=state.pot,
        stack=player.stack,
        hole_class=canonical_hole(player.hole_cards),
        is_pair=ordered[0].rank == ordered[1].rank,
        is_suited=ordered[0].suit == ordered[1].suit,
        is_connected=abs(high - low) == 1,
        high_rank=high,
        low_rank=low,
        board_size=len(state.board),
        pot_odds=pot_odds,
        board_texture=_board_texture(state.board),
        spr=spr,
    )
