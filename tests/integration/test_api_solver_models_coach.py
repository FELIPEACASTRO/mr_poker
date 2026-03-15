from fastapi.testclient import TestClient

from apps.api.main import create_app


def test_solver_model_and_coach_endpoints(tmp_path) -> None:
    app = create_app(db_path=str(tmp_path / 'api_s10.db'))
    client = TestClient(app)

    # create hand and request normalized spot
    created = client.post('/v1/hands/new', json={'seed': 321, 'button_seat': 0, 'stacks': [100, 100]})
    assert created.status_code == 200
    hand_id = created.json()['hand_id']
    spot = client.get(f'/v1/hands/{hand_id}/spots/normalized')
    assert spot.status_code == 200
    assert 'solver_like' in spot.json()

    compare = client.post('/v1/spots/packs/hu_flop_cbet_ip/compare-solver-like')
    assert compare.status_code == 200
    assert 'alignment' in compare.json()

    train = client.post('/v1/models/train/policy-table')
    assert train.status_code == 200
    model_id = train.json()['model_id']
    evaluate = client.post(f'/v1/models/{model_id}/evaluate')
    assert evaluate.status_code == 200
    assert 'alignment_rate' in evaluate.json()

    session = client.post('/v1/sessions/h2h', json={'num_hands': 6, 'seed_base': 4321, 'session_name': 'api_s10'})
    assert session.status_code == 200
    session_id = session.json()['session_id']

    profile = client.get(f'/v1/sessions/{session_id}/opponent-profile')
    assert profile.status_code == 200
    assert 'profiles' in profile.json()

    coach = client.get(f'/v1/sessions/{session_id}/coach-report')
    assert coach.status_code == 200
    assert 'study_plan' in coach.json()

    adaptive = client.post('/v1/benchmark/adaptive', json={'num_hands': 6, 'seed_base': 987})
    assert adaptive.status_code == 200
    assert adaptive.json()['benchmark_type'] == 'adaptive_vs_baseline'
