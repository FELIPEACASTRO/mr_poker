
from packages.engine import GameEngine
from packages.persistence import SqliteHandStore
from services.session_service import SessionConfig, SessionRunner


def test_session_runner_persists_session_and_traces(tmp_path) -> None:
    store = SqliteHandStore(str(tmp_path / 'session.db'))
    runner = SessionRunner(engine=GameEngine(), store=store, report_dir=str(tmp_path / 'reports'))
    payload = runner.run_h2h(SessionConfig(num_hands=6, stacks=(100, 100), seed_base=9000, session_name='test_run'))

    assert payload['hands_played'] == 6
    assert payload['decision_trace_count'] > 0
    assert payload['priced_decision_count'] >= 0
    assert payload['seat_metrics']['seat0']['trace_count'] > 0

    stored = store.get_session(payload['session_id'])
    assert stored is not None
    assert stored['summary']['hands_played'] == 6
    assert len(store.get_decision_traces(session_id=payload['session_id'])) == payload['decision_trace_count']
