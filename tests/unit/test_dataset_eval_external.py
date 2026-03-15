from packages.engine import GameEngine
from packages.persistence import SqliteHandStore
from services.dataset_service import DatasetService
from services.solver_label_service import SolverLabelService
from services.model_service import ModelService
from services.evaluation_service import EvaluationService
from services.external_solver_service import ExternalSolverService
from services.session_service.service import SessionConfig, SessionRunner


def test_dataset_build_and_eval_and_external_solver(tmp_path) -> None:
    engine = GameEngine()
    store = SqliteHandStore(str(tmp_path / 'dataset_eval.db'))
    sessions = SessionRunner(engine=engine, store=store, report_dir=str(tmp_path / 'reports'))
    sessions.run_h2h(SessionConfig(num_hands=6, seed_base=1111, session_name='dataset_seed'))

    solver = SolverLabelService(engine=engine, store=store)
    datasets = DatasetService(store=store, solver_labels=solver, dataset_dir=str(tmp_path / 'datasets'))
    built = datasets.build_master_dataset(dataset_name='test_master')
    assert built['row_count'] >= 6
    assert built['split_counts']['train'] >= 1

    model_service = ModelService(solver_labels=solver, model_dir=str(tmp_path / 'models'))
    trained = model_service.train_policy_table()

    evaluation = EvaluationService(model_registry=model_service.registry, dataset_dir=str(tmp_path / 'datasets'), report_dir=str(tmp_path / 'reports'))
    report = evaluation.evaluate_model(trained['model_id'], built['dataset_id'])
    assert 0.0 <= report['accuracy'] <= 1.0
    assert 'taxonomy_accuracy' in report

    external_dir = tmp_path / 'external_solver'
    external_dir.mkdir(parents=True, exist_ok=True)
    (external_dir / 'sample_solver_v1.json').write_text('{"name":"sample_solver_v1","labels":[{"spot_id":"hu_flop_cbet_ip","action":"bet","confidence":0.8}]}', encoding='utf-8')
    external = ExternalSolverService(solver_labels=solver, base_dir=str(external_dir))
    catalog = external.catalog()
    assert catalog['catalog'][0]['name'] == 'sample_solver_v1'
    comparison = external.compare_spot_pack('sample_solver_v1', 'hu_flop_cbet_ip')
    assert comparison['external_solver']['label']['action'] == 'bet'
