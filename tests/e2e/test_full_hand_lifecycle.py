"""E2E test: Full hand lifecycle from creation through replay."""
from __future__ import annotations

from packages.baseline_agent import BaselineAgent
from packages.common.types import ActionType
from packages.engine.engine import GameEngine
from packages.persistence.sqlite_store import SqliteHandStore


def test_hand_creation_play_persist_replay(tmp_path):
    """Test complete lifecycle: create hand, play it, persist, replay."""
    engine = GameEngine()
    store = SqliteHandStore(str(tmp_path / "e2e.db"))
    agent = BaselineAgent()

    # Create hand
    runtime = engine.start_new_hand(stacks=(100, 100), button_seat=0, seed=42)
    hand_id = runtime.state.hand_id
    initial_snapshot = engine.state_snapshot(runtime)

    store.create_hand(
        hand_id=hand_id,
        button_seat=0,
        stacks=(100, 100),
        seed=42,
        deck_prefix=[],
        initial_snapshot=initial_snapshot,
    )

    # Play hand to completion
    actions_taken = 0
    while not runtime.state.is_terminal:
        decision = agent.decide(runtime, engine)
        actor_seat = runtime.state.acting_seat
        engine.apply_action(runtime, decision.action_type, decision.amount)
        store.append_action(
            hand_id=hand_id,
            actor_seat=int(actor_seat) if actor_seat is not None else -1,
            action_type=decision.action_type.value,
            amount=decision.amount,
        )
        actions_taken += 1

    final_snapshot = engine.state_snapshot(runtime)
    store.append_snapshot(hand_id=hand_id, snapshot=final_snapshot, label="terminal")

    assert runtime.state.is_terminal
    assert actions_taken > 0

    # Verify persistence
    hand = store.get_hand(hand_id)
    assert hand is not None
    assert hand['hand_id'] == hand_id

    persisted_actions = store.get_actions(hand_id)
    assert len(persisted_actions) == actions_taken

    latest = store.get_latest_snapshot(hand_id)
    assert latest is not None
    assert latest['snapshot']['is_terminal']

    # Replay hand
    replay_runtime = engine.start_new_hand(
        stacks=(100, 100),
        button_seat=0,
        hand_id=hand_id,
        seed=42,
        deck_prefix=[],
    )
    for row in persisted_actions:
        engine.apply_action(
            replay_runtime, ActionType(row['action_type']), int(row['amount'])
        )

    import json
    replayed = json.loads(json.dumps(engine.state_snapshot(replay_runtime)))
    persisted = latest['snapshot']
    assert replayed == persisted, "Replayed state should match persisted state"

    # Invariant check
    violations = engine.validate_invariants(replay_runtime)
    assert violations == [], f"Invariant violations: {violations}"


def test_multi_player_hand_lifecycle(tmp_path):
    """Test lifecycle with 4 players."""
    engine = GameEngine()
    store = SqliteHandStore(str(tmp_path / "e2e_multi.db"))
    agent = BaselineAgent()

    runtime = engine.start_new_hand(stacks=(100, 100, 100, 100), button_seat=0, seed=99)
    hand_id = runtime.state.hand_id
    initial_snapshot = engine.state_snapshot(runtime)

    store.create_hand(
        hand_id=hand_id,
        button_seat=0,
        stacks=(100, 100, 100, 100),
        seed=99,
        deck_prefix=[],
        initial_snapshot=initial_snapshot,
    )

    while not runtime.state.is_terminal:
        actions = engine.legal_actions(runtime)
        assert len(actions) > 0
        decision = agent.decide(runtime, engine)
        actor_seat = runtime.state.acting_seat
        engine.apply_action(runtime, decision.action_type, decision.amount)
        store.append_action(
            hand_id=hand_id,
            actor_seat=int(actor_seat) if actor_seat is not None else -1,
            action_type=decision.action_type.value,
            amount=decision.amount,
        )

    assert runtime.state.is_terminal
    total_chips = sum(p.stack for p in runtime.state.players.values()) + runtime.state.pot
    assert total_chips == 400, f"Chip conservation: expected 400, got {total_chips}"
