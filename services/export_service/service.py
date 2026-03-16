from __future__ import annotations

from typing import Any

from packages.exports import export_hand_phh_like, export_session_manifest
from packages.persistence import SqliteHandStore


class ExportService:
    def __init__(self, store: SqliteHandStore) -> None:
        self.store = store

    def export_hand(self, hand_id: str, *, format: str = 'phh_like_v1') -> dict:
        hand = self.store.get_hand(hand_id)
        if hand is None:
            raise KeyError(hand_id)
        latest = self.store.get_latest_snapshot(hand_id)
        if latest is None:
            raise KeyError(hand_id)
        actions = self.store.get_actions(hand_id)
        traces = self.store.get_decision_traces(hand_id=hand_id)

        if format == 'json':
            return {
                'hand_id': hand_id,
                'format': 'json',
                'data': {
                    'hand': hand,
                    'snapshot': latest['snapshot'],
                    'actions': actions,
                    'traces': [t['trace'] for t in traces],
                },
            }

        return {
            'hand_id': hand_id,
            'format': 'phh_like_v1',
            'text': export_hand_phh_like(hand=hand, actions=actions, latest_snapshot=latest['snapshot'], traces=traces),
        }

    def export_session(self, session_id: str) -> dict:
        session = self.store.get_session(session_id)
        if session is None:
            raise KeyError(session_id)
        hands = self.store.get_hands_for_session(session_id)
        return {
            'manifest': export_session_manifest(session=session, hands=hands),
            'hands': [self.export_hand(hand['hand_id']) for hand in hands],
        }

    def export_session_json(self, session_id: str) -> dict[str, Any]:
        """Export session data as structured JSON."""
        session = self.store.get_session(session_id)
        if session is None:
            raise KeyError(session_id)
        hands = self.store.get_hands_for_session(session_id)
        hand_data = []
        for hand in hands:
            hand_data.append(self.export_hand(hand['hand_id'], format='json'))
        return {
            'session_id': session_id,
            'session': session,
            'hands': hand_data,
            'hand_count': len(hand_data),
        }
