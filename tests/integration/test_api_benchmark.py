from fastapi.testclient import TestClient

from apps.api.main import create_app


def test_smoke_benchmark_endpoint(tmp_path) -> None:
    app = create_app(db_path=str(tmp_path / "test.db"))
    client = TestClient(app)
    response = client.post('/v1/benchmark/smoke', json={"num_hands": 8, "seed_base": 500})
    assert response.status_code == 200
    data = response.json()
    assert data["benchmark_type"] == "smoke_agent_vs_agent"
    assert data["hands_played"] == 8
    assert data["seat0_profit"] + data["seat1_profit"] == 0
