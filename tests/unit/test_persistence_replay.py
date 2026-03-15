
from packages.common.types import ActionType
from packages.engine import GameEngine
from packages.persistence import SqliteHandStore
from services.replay_service import ReplayService


def test_persistence_roundtrip_replay(tmp_path) -> None:
    db = tmp_path / 'hands.db'
    store = SqliteHandStore(str(db))
    engine = GameEngine()
    runtime = engine.start_new_hand(stacks=(100, 100), button_seat=0, seed=42)
    initial = engine.state_snapshot(runtime)
    store.create_hand(
        hand_id=runtime.state.hand_id,
        button_seat=0,
        stacks=(100, 100),
        seed=42,
        deck_prefix=[],
        initial_snapshot=initial,
    )
    actor = runtime.state.acting_seat
    engine.apply_action(runtime, ActionType.CALL, 0)
    store.append_action(hand_id=runtime.state.hand_id, actor_seat=int(actor), action_type=ActionType.CALL.value, amount=0)
    store.append_snapshot(hand_id=runtime.state.hand_id, snapshot=engine.state_snapshot(runtime), label='post_action')

    actor = runtime.state.acting_seat
    engine.apply_action(runtime, ActionType.CHECK, 0)
    store.append_action(hand_id=runtime.state.hand_id, actor_seat=int(actor), action_type=ActionType.CHECK.value, amount=0)
    store.append_snapshot(hand_id=runtime.state.hand_id, snapshot=engine.state_snapshot(runtime), label='post_action')

    replay = ReplayService(store=store, engine=GameEngine()).replay_hand(runtime.state.hand_id)
    assert replay['matches_latest_snapshot'] is True
