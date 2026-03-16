import pytest
from fastapi.testclient import TestClient

from packages.engine.engine import GameEngine, HandRuntime
from packages.persistence.sqlite_store import SqliteHandStore


@pytest.fixture
def engine():
    return GameEngine()


@pytest.fixture
def store(tmp_path):
    return SqliteHandStore(str(tmp_path / "test.db"))


@pytest.fixture
def runtime(engine):
    return engine.start_new_hand(stacks=(100, 100), button_seat=0, seed=42)


@pytest.fixture
def test_client(tmp_path):
    """Create a test client with a temporary database."""
    import os
    os.environ["POKER_AUTH_ENABLED"] = "false"
    os.environ["POKER_AI_DB"] = str(tmp_path / "test_api.db")
    from apps.api.main import create_app
    app = create_app(db_path=str(tmp_path / "test_api.db"))
    return TestClient(app)


@pytest.fixture
def test_container(tmp_path):
    """Create a test container with a temporary database."""
    from apps.api.container import build_container
    return build_container(db_path=str(tmp_path / "test_container.db"))


def make_hand_state(
    *,
    stacks: tuple[int, ...] = (100, 100),
    button_seat: int = 0,
    seed: int = 42,
    engine: GameEngine | None = None,
) -> HandRuntime:
    """Factory for creating hand runtimes in tests."""
    eng = engine or GameEngine()
    return eng.start_new_hand(stacks=stacks, button_seat=button_seat, seed=seed)


def make_session(store: SqliteHandStore, engine: GameEngine, *, num_hands: int = 5, seed_base: int = 1000) -> str:
    """Factory for creating a session with hands in tests."""
    from uuid import uuid4
    from packages.baseline_agent import BaselineAgent

    session_id = str(uuid4())
    store.create_session(session_id=session_id, session_type="test", config={"num_hands": num_hands})
    agent = BaselineAgent()

    for i in range(num_hands):
        runtime = engine.start_new_hand(stacks=(100, 100), button_seat=i % 2, seed=seed_base + i)
        snapshot = engine.state_snapshot(runtime)
        store.create_hand(
            hand_id=runtime.state.hand_id,
            session_id=session_id,
            button_seat=i % 2,
            stacks=(100, 100),
            seed=seed_base + i,
            deck_prefix=[],
            initial_snapshot=snapshot,
        )
        while not runtime.state.is_terminal:
            decision = agent.decide(runtime, engine)
            actor_seat = runtime.state.acting_seat
            engine.apply_action(runtime, decision.action_type, decision.amount)
            store.append_action(
                hand_id=runtime.state.hand_id,
                actor_seat=int(actor_seat) if actor_seat is not None else -1,
                action_type=decision.action_type.value,
                amount=decision.amount,
            )
        final_snap = engine.state_snapshot(runtime)
        store.append_snapshot(hand_id=runtime.state.hand_id, snapshot=final_snap, label="terminal")

    store.update_session_summary(session_id=session_id, summary={"hands_played": num_hands})
    return session_id
