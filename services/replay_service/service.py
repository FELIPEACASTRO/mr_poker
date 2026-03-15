
from __future__ import annotations

from packages.common.types import ActionType
from packages.engine.engine import GameEngine
from packages.persistence import SqliteHandStore


class ReplayService:
    def __init__(self, store: SqliteHandStore, engine: GameEngine | None = None) -> None:
        self.store = store
        self.engine = engine or GameEngine()

    def replay_hand(self, hand_id: str) -> dict:
        hand = self.store.get_hand(hand_id)
        if hand is None:
            raise KeyError(hand_id)
        runtime = self.engine.start_new_hand(
            stacks=tuple(hand['stacks']),
            button_seat=hand['button_seat'],
            hand_id=hand_id,
            seed=hand['seed'],
            deck_prefix=list(hand['deck_prefix']),
        )
        for row in self.store.get_actions(hand_id):
            self.engine.apply_action(runtime, ActionType(row['action_type']), int(row['amount']))
        import json
        reconstructed = self.engine.state_snapshot(runtime)
        reconstructed_canonical = json.loads(json.dumps(reconstructed))
        persisted = self.store.get_latest_snapshot(hand_id)
        persisted_snapshot = persisted['snapshot'] if persisted else None
        return {
            'hand_id': hand_id,
            'reconstructed': reconstructed_canonical,
            'persisted_latest': persisted_snapshot,
            'matches_latest_snapshot': reconstructed_canonical == persisted_snapshot,
            'action_count': len(self.store.get_actions(hand_id)),
            'snapshot_count': len(self.store.get_snapshots(hand_id)),
        }
