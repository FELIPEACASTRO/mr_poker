from packages.engine import GameEngine
from packages.persistence import SqliteHandStore
from services.solver_label_service import SolverLabelService
from services.model_service import ModelService
from services.tournament_service import TournamentService
from services.session_service.service import SessionConfig, SessionRunner
from services.curriculum_service import CurriculumService
from services.readiness_service import ReadinessService


def test_tournament_curriculum_and_readiness(tmp_path) -> None:
    engine = GameEngine()
    store = SqliteHandStore(str(tmp_path / 'tour.db'))
    solver = SolverLabelService(engine=engine, store=store)
    model_service = ModelService(solver_labels=solver, model_dir=str(tmp_path / 'models'))
    trained = model_service.train_policy_table()

    tournaments = TournamentService(engine=engine, model_registry=model_service.registry)
    result = tournaments.round_robin(['baseline', 'adaptive', f'policy:{trained["model_id"]}'], hands_per_match=6, seed_base=999)
    assert len(result['leaderboard']) == 3
    assert result['leaderboard'][0]['hands'] >= 6

    sessions = SessionRunner(engine=engine, store=store, report_dir=str(tmp_path / 'reports'))
    session = sessions.run_h2h(SessionConfig(num_hands=8, seed_base=2222, session_name='curriculum'))
    curriculum = CurriculumService(store=store).build_for_session(session['session_id'])
    assert 'curriculum' in curriculum
    assert 'drills' in curriculum['curriculum']

    (tmp_path / 'docs').mkdir()
    (tmp_path / 'tests').mkdir()
    (tmp_path / 'Dockerfile').write_text('FROM scratch', encoding='utf-8')
    dset = tmp_path / 'var' / 'datasets' / 'd1'
    dset.mkdir(parents=True, exist_ok=True)
    (dset / 'manifest.json').write_text('{}', encoding='utf-8')
    readiness = ReadinessService(base_dir=str(tmp_path), dataset_dir='var/datasets', model_dir='models').snapshot()
    assert readiness['docs_ready'] is True
    assert readiness['asset_readiness'] is True
    assert readiness['operational_readiness'] is False
    assert readiness['production_candidate'] is False
    assert readiness['production_candidate'] == readiness['operational_readiness']


def test_readiness_snapshot_exposes_operational_blockers_and_deprecations(tmp_path) -> None:
    (tmp_path / 'docs').mkdir()
    (tmp_path / 'tests').mkdir()
    (tmp_path / 'Dockerfile').write_text('FROM scratch', encoding='utf-8')
    dset = tmp_path / 'var' / 'datasets' / 'd1'
    dset.mkdir(parents=True, exist_ok=True)
    (dset / 'manifest.json').write_text('{}', encoding='utf-8')
    model_dir = tmp_path / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / 'model.json').write_text('{}', encoding='utf-8')
    solver_dir = tmp_path / 'var' / 'external_solver'
    solver_dir.mkdir(parents=True, exist_ok=True)
    (solver_dir / 'sample_solver_v1.json').write_text('{}', encoding='utf-8')

    readiness = ReadinessService(base_dir=str(tmp_path), dataset_dir='var/datasets', model_dir='models').snapshot()
    assert readiness['asset_readiness'] is True
    assert readiness['operational_readiness'] is False
    assert readiness['production_candidate'] == readiness['operational_readiness']
    assert readiness['operational_blockers']
    assert any('sample/mock local mode' in blocker for blocker in readiness['operational_blockers'])
    assert any('production_candidate is deprecated' in item for item in readiness['deprecations'])
