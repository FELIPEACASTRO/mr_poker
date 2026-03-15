from __future__ import annotations

from fastapi.testclient import TestClient

from apps.api.main import create_app


def test_ui_route_serves_index_assets_and_spa_fallback(tmp_path) -> None:
    ui_dist = tmp_path / "dist"
    assets = ui_dist / "assets"
    assets.mkdir(parents=True)
    (ui_dist / "index.html").write_text("<html><body>mr_poker ui</body></html>", encoding="utf-8")
    (assets / "main.js").write_text("console.log('ui');", encoding="utf-8")

    app = create_app(db_path=str(tmp_path / "ui.db"), ui_dist_path=str(ui_dist))
    client = TestClient(app)

    entry = client.get("/ui")
    assert entry.status_code == 200
    assert "mr_poker ui" in entry.text

    asset = client.get("/ui/assets/main.js")
    assert asset.status_code == 200
    assert "console.log('ui');" in asset.text

    fallback = client.get("/ui/models")
    assert fallback.status_code == 200
    assert "mr_poker ui" in fallback.text


def test_ui_addition_does_not_break_existing_api_contracts(tmp_path) -> None:
    ui_dist = tmp_path / "dist"
    ui_dist.mkdir(parents=True)
    (ui_dist / "index.html").write_text("<html><body>mr_poker ui</body></html>", encoding="utf-8")

    app = create_app(db_path=str(tmp_path / "ui_contract.db"), ui_dist_path=str(ui_dist))
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    smoke = client.post("/v1/benchmark/smoke", json={"num_hands": 4, "seed_base": 9000})
    assert smoke.status_code == 200
    smoke_payload = smoke.json()
    assert smoke_payload["benchmark_type"] == "smoke_agent_vs_agent"
    assert smoke_payload["hands_played"] == 4
