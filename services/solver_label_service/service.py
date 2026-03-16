from __future__ import annotations

from typing import Any

from packages.baseline_agent import BaselineAgent
from packages.engine import GameEngine
from packages.persistence import SqliteHandStore
from packages.solver_like import SolverLikeLabeler, bucketize_spot, normalize_runtime
from services.spot_pack_service import SpotPackService


class SolverLabelService:
    def __init__(
        self,
        engine: GameEngine,
        store: SqliteHandStore,
        *,
        labeler: SolverLikeLabeler | None = None,
        baseline: BaselineAgent | None = None,
        spot_packs: SpotPackService | None = None,
    ) -> None:
        self.engine = engine
        self.store = store
        self.labeler = labeler or SolverLikeLabeler()
        self.baseline = baseline or BaselineAgent()
        self.spot_packs = spot_packs or SpotPackService(engine=engine, store=store)

    def normalized_spot_for_runtime(self, runtime) -> dict[str, Any]:
        base = normalize_runtime(runtime, self.engine)
        decision = self.baseline.decide(runtime, self.engine)
        base['estimated_equity'] = round(float(decision.trace.estimated_equity) if decision.trace else 0.0, 4)
        base['baseline_action'] = decision.action_type.value
        base['baseline_rationale'] = decision.rationale
        base['bucket_info'] = bucketize_spot(base)
        base['solver_like'] = self.labeler.label_spot(base)
        return base

    def normalized_spot_for_hand(self, hand_id: str) -> dict[str, Any]:
        snapshots = self.store.get_snapshots(hand_id)
        if not snapshots:
            raise KeyError(hand_id)
        latest = snapshots[-1]['snapshot']
        if latest.get('is_terminal'):
            raise ValueError('hand is terminal; no acting spot to normalize')
        hand = self.store.get_hand(hand_id)
        runtime = self.engine.start_new_hand(
            stacks=tuple(hand['stacks']), button_seat=int(hand['button_seat']), seed=hand['seed'], deck_prefix=hand['deck_prefix'], hand_id=hand_id
        )
        for action in self.store.get_actions(hand_id):
            from packages.common.types import ActionType
            self.engine.apply_action(runtime, ActionType(action['action_type']), int(action['amount']))
        return self.normalized_spot_for_runtime(runtime)

    def compare_spot_pack(self, spot_id: str) -> dict[str, Any]:
        runtime, pack = self.spot_packs.instantiate(spot_id)
        normalized = self.normalized_spot_for_runtime(runtime)
        solver_like = normalized['solver_like']
        baseline_action = normalized['baseline_action']
        return {
            'spot_id': spot_id,
            'pack': pack,
            'normalized_spot': normalized,
            'baseline_action': baseline_action,
            'solver_like_action': solver_like['label_action'],
            'alignment': baseline_action == solver_like['label_action'],
            'solver_like_confidence': solver_like['confidence'],
        }

    def build_training_rows_from_spot_packs(self) -> list[dict[str, Any]]:
        rows = []
        for pack in self.spot_packs.list_packs():
            comparison = self.compare_spot_pack(pack['spot_id'])
            spot = comparison['normalized_spot']
            row = {**spot, **spot['solver_like'], 'source': f"spot_pack:{pack['spot_id']}"}
            rows.append(row)
        return rows
