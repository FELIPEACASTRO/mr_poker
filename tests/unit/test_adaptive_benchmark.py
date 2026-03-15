from packages.engine import GameEngine
from packages.persistence import SqliteHandStore
from services.opponent_profile_service import OpponentProfileService


def test_adaptive_benchmark_runs(tmp_path) -> None:
    engine = GameEngine()
    store = SqliteHandStore(str(tmp_path / 'adaptive.db'))
    result = OpponentProfileService(store=store, engine=engine).adaptive_benchmark(num_hands=8, seed_base=1234)
    assert result['benchmark_type'] == 'adaptive_vs_baseline'
    assert result['hands_played'] == 8
    assert result['seat0_profit'] + result['seat1_profit'] == 0
