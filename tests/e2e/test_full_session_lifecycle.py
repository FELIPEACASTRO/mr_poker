"""E2E test: Full session lifecycle."""
from __future__ import annotations

from packages.engine.engine import GameEngine
from packages.persistence.sqlite_store import SqliteHandStore
from services.session_service import SessionConfig, SessionRunner


def test_session_run_and_query(tmp_path):
    """Run a session, then query its data."""
    engine = GameEngine()
    store = SqliteHandStore(str(tmp_path / "session_e2e.db"))
    runner = SessionRunner(engine=engine, store=store, report_dir=str(tmp_path / "reports"))

    config = SessionConfig(num_hands=10, stacks=(100, 100), seed_base=5000, session_name="e2e_test")
    result = runner.run_h2h(config)

    assert 'session_id' in result
    assert result['hands_played'] == 10
    assert len(result['hand_ids']) == 10

    # Verify session is persisted
    session = store.get_session(result['session_id'])
    assert session is not None

    # Verify hands are persisted
    hands = store.get_hands_for_session(result['session_id'])
    assert len(hands) == 10

    # Verify each hand has actions
    for hand in hands:
        actions = store.get_actions(hand['hand_id'])
        assert len(actions) > 0, f"Hand {hand['hand_id']} should have actions"

    # Verify chip conservation per hand
    for hand in hands:
        latest = store.get_latest_snapshot(hand['hand_id'])
        if latest:
            snap = latest['snapshot']
            total = sum(p['stack'] for p in snap['players'].values()) + snap['pot']
            assert total == 200, f"Hand {hand['hand_id']}: chips={total}, expected 200"


def test_session_analytics(tmp_path):
    """Run session then verify analytics work."""
    engine = GameEngine()
    store = SqliteHandStore(str(tmp_path / "analytics_e2e.db"))
    runner = SessionRunner(engine=engine, store=store, report_dir=str(tmp_path / "reports"))

    config = SessionConfig(num_hands=5, stacks=(100, 100), seed_base=6000)
    result = runner.run_h2h(config)

    from services.analytics_service import SessionAnalyticsService
    analytics = SessionAnalyticsService(store=store)
    report = analytics.session_analytics(result['session_id'])

    assert report['session_id'] == result['session_id']
    assert report['hand_count'] == 5
    assert report['trace_count'] > 0
