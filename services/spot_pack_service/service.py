
from __future__ import annotations

from packages.common.types import ActionType
from packages.engine import GameEngine, HandRuntime
from packages.persistence import SqliteHandStore
from packages.spot_packs import SPOT_PACKS, SpotPackScenario


class SpotPackService:
    def __init__(self, engine: GameEngine, store: SqliteHandStore) -> None:
        self.engine = engine
        self.store = store

    def list_packs(self) -> list[dict]:
        return [self._pack_to_dict(pack) for pack in SPOT_PACKS.values()]

    def instantiate(self, spot_id: str) -> tuple[HandRuntime, dict]:
        if spot_id not in SPOT_PACKS:
            raise KeyError(spot_id)
        pack = SPOT_PACKS[spot_id]
        runtime = self.engine.start_new_hand(
            stacks=pack.stacks,
            button_seat=pack.button_seat,
            deck_prefix=pack.deck_prefix,
        )
        self.store.create_hand(
            hand_id=runtime.state.hand_id,
            button_seat=pack.button_seat,
            stacks=pack.stacks,
            seed=None,
            deck_prefix=pack.deck_prefix,
            initial_snapshot=self.engine.state_snapshot(runtime),
        )
        for action in pack.scripted_actions:
            actor = runtime.state.acting_seat
            self.engine.apply_action(runtime, ActionType(action.action_type), action.amount)
            self.store.append_action(
                hand_id=runtime.state.hand_id,
                actor_seat=int(actor) if actor is not None else -1,
                action_type=action.action_type,
                amount=action.amount,
            )
            self.store.append_snapshot(
                hand_id=runtime.state.hand_id,
                snapshot=self.engine.state_snapshot(runtime),
                label=f'spot_pack:{pack.spot_id}',
            )
        return runtime, self._pack_to_dict(pack)

    def _pack_to_dict(self, pack: SpotPackScenario) -> dict:
        return {
            'spot_id': pack.spot_id,
            'name': pack.name,
            'description': pack.description,
            'objective': pack.objective,
            'stacks': list(pack.stacks),
            'button_seat': pack.button_seat,
            'deck_prefix': pack.deck_prefix,
            'scripted_actions': [action.__dict__ for action in pack.scripted_actions],
            'expected_acting_seat': pack.expected_acting_seat,
            'tags': pack.tags,
        }
