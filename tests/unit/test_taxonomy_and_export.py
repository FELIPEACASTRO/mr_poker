from packages.baseline_agent import BaselineAgent
from packages.engine import GameEngine
from packages.persistence import SqliteHandStore
from services.analytics_service import SessionAnalyticsService
from services.export_service import ExportService
from services.taxonomy_service import TaxonomyService


def test_hand_taxonomy_and_export(tmp_path) -> None:
    engine = GameEngine()
    store = SqliteHandStore(str(tmp_path / 'taxonomy.db'))
    runtime = engine.start_new_hand(stacks=(100, 100), button_seat=0, seed=123)
    store.create_hand(
        hand_id=runtime.state.hand_id,
        button_seat=0,
        stacks=(100, 100),
        seed=123,
        deck_prefix=[],
        initial_snapshot=engine.state_snapshot(runtime),
    )
    agent = BaselineAgent()
    for _ in range(4):
        decision = agent.decide(runtime, engine)
        actor_seat = runtime.state.acting_seat
        engine.apply_action(runtime, decision.action_type, decision.amount)
        store.append_action(hand_id=runtime.state.hand_id, actor_seat=actor_seat, action_type=decision.action_type.value, amount=decision.amount)
        if decision.trace is not None:
            store.append_decision_trace(session_id=None, hand_id=runtime.state.hand_id, actor_seat=decision.trace.actor_seat, trace=decision.trace.model_dump(mode='json'))
        store.append_snapshot(hand_id=runtime.state.hand_id, snapshot=engine.state_snapshot(runtime), label='test_progress')
        if runtime.state.is_terminal:
            break

    taxonomy = TaxonomyService(store).classify_hand(runtime.state.hand_id)
    assert taxonomy['hand_id'] == runtime.state.hand_id
    assert 'taxonomy_tags' in taxonomy

    exported = ExportService(store).export_hand(runtime.state.hand_id)
    assert exported['format'] == 'phh_like_v1'
    assert f'hand_id = "{runtime.state.hand_id}"' in exported['text']


def test_session_analytics_has_taxonomy_and_edges(tmp_path) -> None:
    engine = GameEngine()
    store = SqliteHandStore(str(tmp_path / 'analytics.db'))
    session_id = 'session-1'
    store.create_session(session_id=session_id, session_type='h2h_local', config={'num_hands': 2})
    agent = BaselineAgent()

    for i in range(2):
        runtime = engine.start_new_hand(stacks=(100, 100), button_seat=i % 2, seed=1000 + i)
        store.create_hand(
            hand_id=runtime.state.hand_id,
            session_id=session_id,
            button_seat=i % 2,
            stacks=(100, 100),
            seed=1000 + i,
            deck_prefix=[],
            initial_snapshot=engine.state_snapshot(runtime),
        )
        while not runtime.state.is_terminal:
            decision = agent.decide(runtime, engine)
            seat = runtime.state.acting_seat
            engine.apply_action(runtime, decision.action_type, decision.amount)
            store.append_action(hand_id=runtime.state.hand_id, actor_seat=seat, action_type=decision.action_type.value, amount=decision.amount)
            store.append_snapshot(hand_id=runtime.state.hand_id, snapshot=engine.state_snapshot(runtime), label='session_progress')
            if decision.trace is not None:
                store.append_decision_trace(session_id=session_id, hand_id=runtime.state.hand_id, actor_seat=decision.trace.actor_seat, trace=decision.trace.model_dump(mode='json'))

    analytics = SessionAnalyticsService(store).session_analytics(session_id)
    assert analytics['session_id'] == session_id
    assert analytics['hand_count'] == 2
    assert analytics['trace_count'] > 0
    assert 'taxonomy_counts' in analytics
    assert 'avg_edge_by_street' in analytics
    assert len(analytics['hand_taxonomy']) == 2
