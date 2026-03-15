
from fastapi.testclient import TestClient

from apps.api.main import create_app


def test_api_h2h_and_experiment_run(tmp_path) -> None:
    app = create_app(str(tmp_path / 'api.db'))
    client = TestClient(app)

    h2h = client.post('/v1/benchmark/h2h', json={'num_matches': 2, 'hands_per_match': 10, 'seed_base': 321})
    assert h2h.status_code == 200
    body = h2h.json()
    assert body['summary']['total_hands'] == 20

    exp = client.post('/v1/experiments/run', json={'num_matches': 2, 'hands_per_match': 10, 'seed_base': 654})
    assert exp.status_code == 200
    assert 'report_path' in exp.json()
