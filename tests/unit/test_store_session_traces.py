
from packages.persistence import SqliteHandStore


def test_store_sessions_and_traces(tmp_path) -> None:
    store = SqliteHandStore(str(tmp_path / 'store.db'))
    store.create_session(session_id='s1', session_type='h2h_local', config={'num_hands': 2})
    store.create_hand(
        hand_id='h1', button_seat=0, stacks=(100, 100), seed=1,
        deck_prefix=[], initial_snapshot={}, session_id='s1',
    )
    store.append_decision_trace(session_id='s1', hand_id='h1', actor_seat=0, trace={'foo': 'bar'})
    store.update_session_summary(session_id='s1', summary={'ok': True})

    session = store.get_session('s1')
    assert session is not None
    assert session['config']['num_hands'] == 2
    assert session['summary']['ok'] is True

    traces = store.get_decision_traces(session_id='s1')
    assert len(traces) == 1
    assert traces[0]['trace']['foo'] == 'bar'
