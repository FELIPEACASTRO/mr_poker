from fastapi.testclient import TestClient

from apps.api.main import create_app


def test_api_sprint20_dataset_tournament_and_readiness(tmp_path) -> None:
    app = create_app(db_path=str(tmp_path / 'api20.db'))
    client = TestClient(app)

    session = client.post('/v1/sessions/h2h', json={'num_hands': 6, 'seed_base': 8080, 'session_name': 'api20'})
    assert session.status_code == 200
    session_id = session.json()['session_id']

    build = client.post('/v1/datasets/build', params={'dataset_name': 'api20_master'})
    assert build.status_code == 200
    dataset_id = build.json()['dataset_id']

    train = client.post('/v1/models/train/policy-table')
    assert train.status_code == 200
    model_id = train.json()['model_id']

    evaluate = client.post(f'/v1/models/{model_id}/evaluate-dataset/{dataset_id}')
    assert evaluate.status_code == 200
    assert 'accuracy' in evaluate.json()

    catalog = client.get('/v1/external-solver/catalog')
    assert catalog.status_code == 200
    assert catalog.json()['catalog'][0]['name'] == 'sample_solver_v1'

    ext = client.post('/v1/external-solver/sample_solver_v1/compare/hu_flop_cbet_ip')
    assert ext.status_code == 200
    assert ext.json()['external_solver']['label']['action'] == 'bet'

    tour = client.post('/v1/tournaments/round-robin', json={'entrants': ['baseline', 'adaptive'], 'hands_per_match': 6})
    assert tour.status_code == 200
    assert len(tour.json()['leaderboard']) == 2

    curriculum = client.get(f'/v1/sessions/{session_id}/curriculum')
    assert curriculum.status_code == 200
    assert 'curriculum' in curriculum.json()

    readiness = client.get('/v1/system/readiness')
    assert readiness.status_code == 200
    payload = readiness.json()
    assert 'production_candidate' in payload
    assert 'asset_readiness' in payload
    assert 'operational_readiness' in payload
    assert 'operational_blockers' in payload
    assert 'deprecations' in payload
