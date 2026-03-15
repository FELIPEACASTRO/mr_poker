from __future__ import annotations

from packages.persistence import SqliteHandStore
from packages.taxonomy import TAXONOMY_CATALOG, classify_hand


class TaxonomyService:
    def __init__(self, store: SqliteHandStore) -> None:
        self.store = store

    def catalog(self) -> dict:
        return {'catalog': TAXONOMY_CATALOG}

    def classify_hand(self, hand_id: str) -> dict:
        latest = self.store.get_latest_snapshot(hand_id)
        if latest is None:
            raise KeyError(hand_id)
        traces = self.store.get_decision_traces(hand_id=hand_id)
        actions = self.store.get_actions(hand_id)
        return classify_hand(latest['snapshot'], [t['trace'] for t in traces], actions)
