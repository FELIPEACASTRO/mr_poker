from fastapi.testclient import TestClient

from apps.api.main import create_app


def test_api_taxonomy_export_and_session_analytics(tmp_path) -> None:
    app = create_app(str(tmp_path / 'api6.db'))
    client = TestClient(app)

    # Hand lifecycle for export and taxonomy.
    created = client.post('/v1/hands/new', json={'stacks': [100, 100], 'button_seat': 0, 'seed': 707})
    assert created.status_code == 200
    hand_id = created.json()['hand_id']

    for _ in range(4):
        acted = client.post(f'/v1/hands/{hand_id}/auto')
        assert acted.status_code == 200
        if acted.json()['is_terminal']:
            break

    taxonomy = client.get(f'/v1/hands/{hand_id}/taxonomy')
    assert taxonomy.status_code == 200
    assert taxonomy.json()['hand_id'] == hand_id

    exported = client.get(f'/v1/hands/{hand_id}/export/phh-like')
    assert exported.status_code == 200
    assert exported.json()['format'] == 'phh_like_v1'
    assert hand_id in exported.json()['text']

    catalog = client.get('/v1/spots/taxonomy')
    assert catalog.status_code == 200
    assert 'catalog' in catalog.json()

    session = client.post('/v1/sessions/h2h', json={'num_hands': 6, 'stacks': [100, 100], 'seed_base': 16000, 'session_name': 'sprint06_session'})
    assert session.status_code == 200
    session_id = session.json()['session_id']

    analytics = client.get(f'/v1/sessions/{session_id}/analytics')
    assert analytics.status_code == 200
    body = analytics.json()
    assert body['session_id'] == session_id
    assert body['trace_count'] > 0
    assert body['hand_count'] == 6

    session_export = client.get(f'/v1/sessions/{session_id}/export/phh-like')
    assert session_export.status_code == 200
    export_body = session_export.json()
    assert export_body['manifest']['session_id'] == session_id
    assert export_body['manifest']['hand_count'] == 6
    assert len(export_body['hands']) == 6
