
from fastapi.testclient import TestClient

from apps.api.main import create_app


def _bootstrap_model_and_dataset(client: TestClient):
    ds = client.post('/v1/datasets/build', params={'dataset_name': 'master_v1'}).json()
    client.post('/v1/models/train/policy-table')
    return ds['dataset_id']


def test_sprint30_endpoints(tmp_path):
    app = create_app(db_path=str(tmp_path / 'api30.db'))
    client = TestClient(app)
    dataset_id = _bootstrap_model_and_dataset(client)

    models = client.get('/v1/models').json()['models']
    model_id = models[0]['model_id']

    gov = client.post(f'/v1/governance/model-cards/{model_id}')
    assert gov.status_code == 200
    assert gov.json()['model_id'] == model_id

    gate = client.post(f'/v1/release/gate/{model_id}/{dataset_id}', json={'min_accuracy': 0.0, 'min_rows': 1})
    assert gate.status_code == 200
    assert 'go_for_alpha' in gate.json()

    batch = client.post('/v1/hands/batch-generate', json={'batch_name': 'b1', 'num_hands': 4, 'seed_base': 123000, 'stacks': [100, 100]})
    assert batch.status_code == 200
    assert batch.json()['hands_played'] == 4

    suite = client.post('/v1/regression/suites/build', json={'suite_name': 'suite_api'})
    assert suite.status_code == 200
    assert suite.json()['suite_name'] == 'suite_api'

    cal = client.post(f'/v1/models/{model_id}/calibrate/{dataset_id}')
    assert cal.status_code == 200
    assert 'ece_proxy' in cal.json()

    deploy = client.get('/v1/deploy/manifest')
    assert deploy.status_code == 200
    assert deploy.json()['dockerfile_present'] is True

    notes = client.get('/v1/release/notes')
    assert notes.status_code == 200
    assert notes.json()['release_train'] == 'Sprint 21-30'

    alpha = client.get('/v1/system/alpha-candidate', params={'model_id': model_id, 'dataset_id': dataset_id})
    assert alpha.status_code == 200
    assert 'alpha_candidate' in alpha.json()
