
from fastapi.testclient import TestClient

from apps.api.main import create_app


def test_api_spot_pack_and_session_endpoints(tmp_path) -> None:
    app = create_app(str(tmp_path / 'api5.db'))
    client = TestClient(app)

    listed = client.get('/v1/spots/packs')
    assert listed.status_code == 200
    packs = listed.json()['packs']
    assert any(pack['spot_id'] == 'hu_flop_cbet_ip' for pack in packs)

    instantiated = client.post('/v1/spots/packs/hu_flop_cbet_ip/instantiate')
    assert instantiated.status_code == 200
    body = instantiated.json()
    assert body['snapshot']['street'] == 'flop'
    assert body['snapshot']['acting_seat'] == 0

    session = client.post('/v1/sessions/h2h', json={'num_hands': 8, 'stacks': [100, 100], 'seed_base': 15000, 'session_name': 'api_session'})
    assert session.status_code == 200
    payload = session.json()
    assert payload['hands_played'] == 8
    assert payload['decision_trace_count'] > 0

    session_view = client.get(f"/v1/sessions/{payload['session_id']}")
    assert session_view.status_code == 200
    assert session_view.json()['summary']['hands_played'] == 8

    traces = client.get(f"/v1/sessions/{payload['session_id']}/traces")
    assert traces.status_code == 200
    assert len(traces.json()['traces']) == payload['decision_trace_count']
