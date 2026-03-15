from fastapi.testclient import TestClient

from apps.api.main import app


client = TestClient(app)


def test_create_hand_and_fetch_it() -> None:
    created = client.post("/v1/hands/new", json={"stacks": [100, 100], "seed": 11})
    assert created.status_code == 200
    payload = created.json()
    assert payload["pot"] == 3
    hand_id = payload["hand_id"]

    fetched = client.get(f"/v1/hands/{hand_id}")
    assert fetched.status_code == 200
    assert fetched.json()["hand_id"] == hand_id
