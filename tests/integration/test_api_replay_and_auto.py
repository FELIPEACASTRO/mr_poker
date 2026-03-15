
from fastapi.testclient import TestClient

from apps.api.main import create_app


def test_api_auto_and_replay(tmp_path) -> None:
    app = create_app(str(tmp_path / 'api.db'))
    client = TestClient(app)

    created = client.post('/v1/hands/new', json={'stacks': [100, 100], 'button_seat': 0, 'seed': 7})
    assert created.status_code == 200
    hand = created.json()
    hand_id = hand['hand_id']

    auto = client.post(f'/v1/hands/{hand_id}/auto')
    assert auto.status_code == 200
    assert 'decision_rationale' in auto.json()
    assert 'decision_trace' in auto.json()

    replay = client.get(f'/v1/hands/{hand_id}/replay')
    assert replay.status_code == 200
    body = replay.json()
    assert body['matches_latest_snapshot'] is True
