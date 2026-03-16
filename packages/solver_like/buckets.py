from __future__ import annotations

from typing import Any


def _spr_bucket(spr: float) -> str:
    if spr < 2:
        return 'spr_lt_2'
    if spr < 4:
        return 'spr_2_4'
    if spr < 8:
        return 'spr_4_8'
    return 'spr_ge_8'


def _equity_bucket(equity: float) -> str:
    if equity < 0.35:
        return 'eq_low'
    if equity < 0.5:
        return 'eq_medium'
    if equity < 0.65:
        return 'eq_good'
    return 'eq_strong'


def _price_bucket(edge: float) -> str:
    if edge < -0.08:
        return 'underpriced_bad'
    if edge < 0.0:
        return 'underpriced_close'
    if edge < 0.08:
        return 'priced_close'
    return 'priced_good'


def bucketize_spot(spot: dict[str, Any]) -> dict[str, str]:
    equity = float(spot.get('estimated_equity', 0.0))
    pot_odds = float(spot.get('pot_odds', 0.0))
    edge = equity - pot_odds
    tags = {
        'street_bucket': str(spot.get('street', 'unknown')),
        'hole_bucket': str(spot.get('hole_class', '??')),
        'texture_bucket': str(spot.get('board_texture', 'unknown')),
        'spr_bucket': _spr_bucket(float(spot.get('spr', 0.0))),
        'equity_bucket': _equity_bucket(equity),
        'price_bucket': _price_bucket(edge),
        'position_bucket': 'ip' if int(spot.get('actor_seat', 0)) == int(spot.get('button_seat', 0)) else 'oop',
    }
    tags['bucket_key'] = '|'.join([
        tags['street_bucket'], tags['position_bucket'], tags['spr_bucket'],
        tags['equity_bucket'], tags['price_bucket'], tags['texture_bucket'],
    ])
    return tags
