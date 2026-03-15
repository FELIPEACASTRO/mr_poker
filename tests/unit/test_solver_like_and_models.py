from packages.engine import GameEngine
from packages.persistence import SqliteHandStore
from services.solver_label_service import SolverLabelService
from services.model_service import ModelService


def test_solver_like_compare_and_policy_training(tmp_path) -> None:
    engine = GameEngine()
    store = SqliteHandStore(str(tmp_path / 'solver_like.db'))
    solver = SolverLabelService(engine=engine, store=store)

    comparison = solver.compare_spot_pack('hu_flop_cbet_ip')
    assert comparison['spot_id'] == 'hu_flop_cbet_ip'
    assert 'normalized_spot' in comparison
    assert comparison['solver_like_action'] in {'bet', 'check', 'call', 'raise', 'fold', 'all_in'}

    model_service = ModelService(solver_labels=solver, model_dir=str(tmp_path / 'models'))
    trained = model_service.train_policy_table()
    assert trained['training_rows'] >= 1
    models = model_service.list_models()
    assert len(models) >= 1
    evaluation = model_service.evaluate_model_on_spot_packs(trained['model_id'])
    assert evaluation['rows'] >= 1
    assert 0.0 <= evaluation['alignment_rate'] <= 1.0
