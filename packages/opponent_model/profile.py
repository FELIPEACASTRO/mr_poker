from __future__ import annotations

from collections import Counter
from typing import Any


AGGRESSIVE = {'bet', 'raise', 'all_in'}
VOLUNTARY = {'call', 'bet', 'raise', 'all_in'}


def build_opponent_profile(actions: list[dict[str, Any]], *, seat: int) -> dict[str, Any]:
    seat_actions = [a for a in actions if int(a.get('actor_seat', a.get('seat', -1))) == seat]
    preflop = [a for a in seat_actions if str(a.get('street', 'pre_flop')) == 'pre_flop']
    voluntary = [a for a in preflop if str(a.get('action_type', a.get('action', ''))) in VOLUNTARY]
    pfr = [a for a in preflop if str(a.get('action_type', a.get('action', ''))) in {'bet', 'raise', 'all_in'}]
    aggressive = [a for a in seat_actions if str(a.get('action_type', a.get('action', ''))) in AGGRESSIVE]
    counts = Counter(str(a.get('action_type', a.get('action', ''))) for a in seat_actions)
    denom = max(1, len(preflop))
    return {
        'seat': seat,
        'action_count': len(seat_actions),
        'action_mix': dict(counts),
        'vpip_proxy': round(len(voluntary) / denom, 4),
        'pfr_proxy': round(len(pfr) / denom, 4),
        'aggression_rate': round(len(aggressive) / max(1, len(seat_actions)), 4),
        'style_tag': summarize_profile({
            'vpip_proxy': len(voluntary) / denom,
            'pfr_proxy': len(pfr) / denom,
            'aggression_rate': len(aggressive) / max(1, len(seat_actions)),
        }),
    }


def summarize_profile(profile: dict[str, Any]) -> str:
    vpip = float(profile.get('vpip_proxy', 0.0))
    pfr = float(profile.get('pfr_proxy', 0.0))
    agg = float(profile.get('aggression_rate', 0.0))
    if vpip >= 0.55 and agg >= 0.45:
        return 'loose_aggressive'
    if vpip >= 0.55 and agg < 0.45:
        return 'loose_passive'
    if vpip < 0.35 and pfr >= 0.20:
        return 'tight_aggressive'
    return 'tight_passive'
