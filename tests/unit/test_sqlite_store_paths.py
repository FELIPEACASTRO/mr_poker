from __future__ import annotations

import sqlite3

from packages.persistence import SqliteHandStore


def test_sqlite_store_migrates_legacy_hands_schema(tmp_path) -> None:
    db_path = tmp_path / "legacy_hands.db"
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("""
            CREATE TABLE hands (
                hand_id TEXT PRIMARY KEY,
                button_seat INTEGER NOT NULL,
                stacks_json TEXT NOT NULL,
                seed INTEGER,
                deck_prefix_json TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """)

    SqliteHandStore(str(db_path))
    with sqlite3.connect(str(db_path)) as conn:
        cols = {
            str(row[1]) for row in conn.execute("PRAGMA table_info(hands)").fetchall()
        }
    assert "session_id" in cols


def test_store_none_paths_and_explicit_action_order(tmp_path) -> None:
    store = SqliteHandStore(str(tmp_path / "store_none_paths.db"))

    assert store.get_session("missing-session") is None
    assert store.get_hand("missing-hand") is None

    store.create_hand(
        hand_id="h-explicit",
        button_seat=0,
        stacks=(100, 100),
        seed=42,
        deck_prefix=[],
        initial_snapshot={"pot": 3},
        session_id="s-explicit",
    )
    store.append_action_with_order(
        hand_id="h-explicit",
        actor_seat=0,
        action_type="call",
        amount=2,
        action_order=7,
    )
    actions = store.get_actions("h-explicit")
    assert len(actions) == 1
    assert int(actions[0]["action_order"]) == 7


def test_delete_session_data_cascades_all_related_rows(tmp_path) -> None:
    store = SqliteHandStore(str(tmp_path / "store_delete_session.db"))

    store.create_session(
        session_id="s-delete",
        session_type="h2h_local",
        config={"num_hands": 2},
    )
    store.create_hand(
        hand_id="h1",
        button_seat=0,
        stacks=(100, 100),
        seed=1,
        deck_prefix=[],
        initial_snapshot={"pot": 3},
        session_id="s-delete",
    )
    store.create_hand(
        hand_id="h2",
        button_seat=1,
        stacks=(100, 100),
        seed=2,
        deck_prefix=[],
        initial_snapshot={"pot": 3},
        session_id="s-delete",
    )
    store.append_action(hand_id="h1", actor_seat=0, action_type="call", amount=2)
    store.append_action(hand_id="h2", actor_seat=1, action_type="check", amount=0)
    store.append_decision_trace(
        session_id="s-delete",
        hand_id="h1",
        actor_seat=0,
        trace={"policy": "aggressive"},
    )
    store.append_decision_trace(
        session_id="s-delete",
        hand_id="h2",
        actor_seat=1,
        trace={"policy": "defensive"},
    )

    store.delete_session_data(session_id="s-delete")

    assert store.get_session("s-delete") is None
    assert store.get_hands_for_session("s-delete") == []
    assert store.get_decision_traces(session_id="s-delete") == []
    assert store.get_hand("h1") is None
    assert store.get_hand("h2") is None


def test_uow_rolls_back_when_step_fails(tmp_path) -> None:
    store = SqliteHandStore(str(tmp_path / "store_uow_rollback.db"))
    store.create_session(
        session_id="s-uow",
        session_type="h2h_local",
        config={"num_hands": 1},
    )

    try:
        with store.uow() as uow:
            uow.create_hand(
                hand_id="h-uow",
                button_seat=0,
                stacks=(100, 100),
                seed=7,
                deck_prefix=[],
                initial_snapshot={"pot": 3},
                session_id="s-uow",
            )
            uow.append_hand_batch(
                hand_id="h-uow",
                session_id="s-uow",
                actions=[(0, "call", 2, 0)],
                snapshots=[(1, {"pot": 5}, "post_action")],
                decision_traces=[],
            )
            raise RuntimeError("force rollback")
    except RuntimeError:
        pass

    assert store.get_hand("h-uow") is None
    assert store.get_actions("h-uow") == []
    assert store.get_snapshots("h-uow") == []
