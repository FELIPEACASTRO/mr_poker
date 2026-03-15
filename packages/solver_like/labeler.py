from __future__ import annotations

from typing import Any

from packages.common.types import ActionType
from .buckets import bucketize_spot


class SolverLikeLabeler:
    """Local heuristic labeler used until a true solver pipeline is integrated.

    It is intentionally conservative and produces reproducible pseudo-solver labels.
    """

    def label_spot(self, spot: dict[str, Any]) -> dict[str, Any]:
        legal = set(str(a) for a in spot.get('legal_actions', []))
        equity = float(spot.get('estimated_equity', 0.0))
        pot_odds = float(spot.get('pot_odds', 0.0))
        edge = equity - pot_odds
        street = str(spot.get('street', 'pre_flop'))
        to_call = int(spot.get('to_call', 0))
        spr = float(spot.get('spr', 0.0))
        bucket_info = bucketize_spot(spot)

        if to_call == 0:
            if equity >= 0.72 and 'all_in' in legal and spr < 2:
                action = ActionType.ALL_IN.value
                reason = 'short SPR jam region'
            elif equity >= 0.62 and 'bet' in legal:
                action = ActionType.BET.value
                reason = 'value/protection betting region'
            else:
                action = ActionType.CHECK.value if 'check' in legal else sorted(legal)[0]
                reason = 'check-back or fallback region'
        else:
            if equity >= max(0.78, pot_odds + 0.22) and 'raise' in legal:
                action = ActionType.RAISE.value
                reason = 'clear raise value edge'
            elif equity >= max(0.64, pot_odds + 0.08) and 'call' in legal:
                action = ActionType.CALL.value
                reason = 'continue profitable versus price'
            elif equity >= 0.68 and 'all_in' in legal and spr <= 1.5:
                action = ActionType.ALL_IN.value
                reason = 'jam with low SPR edge'
            else:
                action = ActionType.FOLD.value if 'fold' in legal else ('check' if 'check' in legal else sorted(legal)[0])
                reason = 'fold/check disciplined region'

        confidence = min(0.98, max(0.51, 0.6 + abs(edge) + (0.08 if street == 'river' else 0.0)))
        return {
            'label_action': action,
            'label_amount_rule': 'solver_like_v1_rule',
            'label_reason': reason,
            'confidence': round(confidence, 4),
            'bucket_info': bucket_info,
        }
