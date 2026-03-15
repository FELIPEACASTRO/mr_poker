from __future__ import annotations

import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from packages.baseline_agent import BaselineAgent
from packages.common.types import ActionType
from packages.engine import GameEngine, HandRuntime
from packages.persistence import SqliteHandStore
from services.analytics_service import SessionAnalyticsService
from services.benchmark_service import BenchmarkService
from services.experiment_runner import ExperimentRunner
from services.export_service import ExportService
from services.solver_label_service import SolverLabelService
from services.model_service import ModelService
from services.opponent_profile_service import OpponentProfileService
from services.coach_service import CoachService
from services.replay_service.service import ReplayService
from services.session_service import SessionConfig, SessionRunner
from services.spot_pack_service import SpotPackService
from services.taxonomy_service import TaxonomyService
from services.dataset_service import DatasetService
from services.evaluation_service import EvaluationService
from services.external_solver_service import ExternalSolverService
from services.tournament_service import TournamentService
from services.curriculum_service import CurriculumService
from services.readiness_service import ReadinessService
from services.governance_service import GovernanceService
from services.release_gate_service import ReleaseGateService
from services.batch_generation_service import BatchGenerationService
from services.regression_suite_service import RegressionSuiteService
from services.calibration_service import CalibrationService
from services.deploy_service import DeployService
from services.release_notes_service import ReleaseNotesService
from services.alpha_candidate_service import AlphaCandidateService


class NewHandRequest(BaseModel):
    stacks: tuple[int, int] = (100, 100)
    button_seat: int = 0
    seed: Optional[int] = None
    deck_prefix: list[str] = Field(default_factory=list)


class ActionRequest(BaseModel):
    action_type: ActionType
    amount: int = 0


class SmokeBenchmarkRequest(BaseModel):
    num_hands: int = 50
    stacks: tuple[int, int] = (100, 100)
    seed_base: int = 10_000


class H2HBenchmarkRequest(BaseModel):
    num_matches: int = 3
    hands_per_match: int = 50
    stacks: tuple[int, int] = (100, 100)
    seed_base: int = 20_000


class SessionRunRequest(BaseModel):
    num_hands: int = 100
    stacks: tuple[int, int] = (100, 100)
    seed_base: int = 30_000
    session_name: str = 'local_h2h_session'


class AdaptiveBenchmarkRequest(BaseModel):
    num_hands: int = 40
    stacks: tuple[int, int] = (100, 100)
    seed_base: int = 50_000


class TournamentRequest(BaseModel):
    entrants: list[str] = Field(default_factory=lambda: ['baseline', 'adaptive'])
    hands_per_match: int = 20
    seed_base: int = 70_000
    stacks: tuple[int, int] = (100, 100)


class BatchGenerateRequest(BaseModel):
    batch_name: str = 'baseline_batch'
    num_hands: int = 50
    seed_base: int = 90_000
    stacks: tuple[int, int] = (100, 100)


class ReleaseGateRequest(BaseModel):
    min_accuracy: float = 0.55
    min_rows: int = 10


class RegressionSuiteRequest(BaseModel):
    suite_name: str = 'core_regression_v1'


def create_app(db_path: str | None = None) -> FastAPI:
    app = FastAPI(title='Poker AI Local API', version='0.30.0')
    engine = GameEngine()
    agent = BaselineAgent()
    store = SqliteHandStore(db_path or os.getenv('POKER_AI_DB', 'var/poker_ai_local.db'))
    replay = ReplayService(store=store, engine=engine)
    benchmark = BenchmarkService(engine=engine)
    experiments = ExperimentRunner(engine=engine)
    sessions = SessionRunner(engine=engine, store=store)
    spot_packs = SpotPackService(engine=engine, store=store)
    taxonomy = TaxonomyService(store=store)
    analytics = SessionAnalyticsService(store=store)
    exports = ExportService(store=store)
    solver_labels = SolverLabelService(engine=engine, store=store)
    models = ModelService(solver_labels=solver_labels)
    opponent_profiles = OpponentProfileService(store=store, engine=engine)
    coach = CoachService(store=store)
    datasets = DatasetService(store=store, solver_labels=solver_labels)
    evaluation = EvaluationService(model_registry=models.registry)
    external_solver = ExternalSolverService(solver_labels=solver_labels)
    tournaments = TournamentService(engine=engine, model_registry=models.registry)
    curriculum = CurriculumService(store=store)
    readiness = ReadinessService(base_dir='.')
    governance = GovernanceService()
    release_gate = ReleaseGateService(readiness=readiness, evaluation=evaluation)
    batch_generation = BatchGenerationService(session_runner=sessions)
    regression_suites = RegressionSuiteService()
    calibration = CalibrationService(evaluation=evaluation)
    deploy = DeployService(base_dir='.')
    release_notes = ReleaseNotesService(docs_dir='docs')
    alpha_candidate = AlphaCandidateService(release_gate=release_gate, governance=governance, deploy=deploy)
    runtimes: dict[str, HandRuntime] = {}

    @app.get('/health')
    def health() -> dict[str, str]:
        return {'status': 'ok', 'version': app.version, 'db_path': store.db_path}

    @app.get('/')
    def root() -> dict[str, str]:
        return {'service': app.title, 'version': app.version, 'docs': '/docs'}

    @app.post('/v1/hands/new')
    def new_hand(payload: NewHandRequest) -> dict:
        runtime = engine.start_new_hand(
            stacks=payload.stacks,
            button_seat=payload.button_seat,
            seed=payload.seed,
            deck_prefix=payload.deck_prefix,
        )
        runtimes[runtime.state.hand_id] = runtime
        snapshot = engine.state_snapshot(runtime)
        store.create_hand(
            hand_id=runtime.state.hand_id,
            button_seat=payload.button_seat,
            stacks=payload.stacks,
            seed=payload.seed,
            deck_prefix=payload.deck_prefix,
            initial_snapshot=snapshot,
        )
        return snapshot

    @app.get('/v1/hands/{hand_id}')
    def get_hand(hand_id: str) -> dict:
        runtime = runtimes.get(hand_id)
        if runtime is not None:
            return engine.state_snapshot(runtime)
        latest = store.get_latest_snapshot(hand_id)
        if latest is None:
            raise HTTPException(status_code=404, detail='hand not found')
        return latest['snapshot']

    @app.post('/v1/hands/{hand_id}/actions')
    def act(hand_id: str, payload: ActionRequest) -> dict:
        runtime = runtimes.get(hand_id)
        if runtime is None:
            raise HTTPException(status_code=404, detail='hand not found')
        try:
            actor_seat = runtime.state.acting_seat
            engine.apply_action(runtime, payload.action_type, payload.amount)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        store.append_action(
            hand_id=hand_id,
            actor_seat=int(actor_seat) if actor_seat is not None else -1,
            action_type=payload.action_type.value,
            amount=payload.amount,
        )
        snapshot = engine.state_snapshot(runtime)
        store.append_snapshot(hand_id=hand_id, snapshot=snapshot, label='post_action')
        return snapshot

    @app.post('/v1/hands/{hand_id}/auto')
    def auto_act(hand_id: str) -> dict:
        runtime = runtimes.get(hand_id)
        if runtime is None:
            raise HTTPException(status_code=404, detail='hand not found')
        try:
            decision = agent.decide(runtime, engine)
            actor_seat = runtime.state.acting_seat
            engine.apply_action(runtime, decision.action_type, decision.amount)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        store.append_action(
            hand_id=hand_id,
            actor_seat=int(actor_seat) if actor_seat is not None else -1,
            action_type=decision.action_type.value,
            amount=decision.amount,
        )
        if decision.trace is not None:
            store.append_decision_trace(
                session_id=None,
                hand_id=hand_id,
                actor_seat=decision.trace.actor_seat,
                trace=decision.trace.model_dump(mode='json'),
            )
        snapshot = engine.state_snapshot(runtime)
        store.append_snapshot(hand_id=hand_id, snapshot=snapshot, label='post_auto_action')
        response = dict(snapshot)
        response['decision_rationale'] = decision.rationale
        response['decision_trace'] = decision.trace.model_dump(mode='json') if decision.trace else None
        response['persisted_trace_count'] = len(store.get_decision_traces(hand_id=hand_id))
        return response

    @app.get('/v1/hands/{hand_id}/traces')
    def get_hand_traces(hand_id: str) -> dict:
        return {'hand_id': hand_id, 'traces': store.get_decision_traces(hand_id=hand_id)}

    @app.get('/v1/hands/{hand_id}/replay')
    def replay_hand(hand_id: str) -> dict:
        try:
            return replay.replay_hand(hand_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail='hand not found') from exc

    @app.get('/v1/hands/{hand_id}/taxonomy')
    def get_hand_taxonomy(hand_id: str) -> dict:
        try:
            return taxonomy.classify_hand(hand_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail='hand not found') from exc

    @app.get('/v1/hands/{hand_id}/export/phh-like')
    def export_hand(hand_id: str) -> dict:
        try:
            return exports.export_hand(hand_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail='hand not found') from exc

    @app.get('/v1/spots/packs')
    def list_spot_packs() -> dict:
        return {'packs': spot_packs.list_packs()}

    @app.get('/v1/spots/taxonomy')
    def taxonomy_catalog() -> dict:
        return taxonomy.catalog()

    @app.post('/v1/spots/packs/{spot_id}/instantiate')
    def instantiate_spot_pack(spot_id: str) -> dict:
        try:
            runtime, pack = spot_packs.instantiate(spot_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail='spot pack not found') from exc
        runtimes[runtime.state.hand_id] = runtime
        snapshot = engine.state_snapshot(runtime)
        return {'pack': pack, 'snapshot': snapshot}

    @app.post('/v1/benchmark/smoke')
    def smoke_benchmark(payload: SmokeBenchmarkRequest) -> dict:
        return benchmark.smoke(num_hands=payload.num_hands, stacks=payload.stacks, seed_base=payload.seed_base)

    @app.post('/v1/benchmark/h2h')
    def h2h_benchmark(payload: H2HBenchmarkRequest) -> dict:
        return benchmark.h2h_eval(
            num_matches=payload.num_matches,
            hands_per_match=payload.hands_per_match,
            stacks=payload.stacks,
            seed_base=payload.seed_base,
        )

    @app.post('/v1/experiments/run')
    def run_experiment(payload: H2HBenchmarkRequest) -> dict:
        from services.experiment_runner.service import ExperimentConfig
        return experiments.run_h2h(ExperimentConfig(
            num_matches=payload.num_matches,
            hands_per_match=payload.hands_per_match,
            stacks=payload.stacks,
            seed_base=payload.seed_base,
        ))

    @app.post('/v1/sessions/h2h')
    def run_session(payload: SessionRunRequest) -> dict:
        return sessions.run_h2h(SessionConfig(
            num_hands=payload.num_hands,
            stacks=payload.stacks,
            seed_base=payload.seed_base,
            session_name=payload.session_name,
        ))

    @app.get('/v1/sessions/{session_id}')
    def get_session(session_id: str) -> dict:
        result = store.get_session(session_id)
        if result is None:
            raise HTTPException(status_code=404, detail='session not found')
        return result

    @app.get('/v1/sessions/{session_id}/traces')
    def get_session_traces(session_id: str) -> dict:
        traces = store.get_decision_traces(session_id=session_id)
        if not traces:
            session = store.get_session(session_id)
            if session is None:
                raise HTTPException(status_code=404, detail='session not found')
        return {'session_id': session_id, 'traces': traces}

    @app.get('/v1/sessions/{session_id}/analytics')
    def get_session_analytics(session_id: str) -> dict:
        try:
            return analytics.session_analytics(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail='session not found') from exc

    @app.get('/v1/sessions/{session_id}/export/phh-like')
    def export_session(session_id: str) -> dict:
        try:
            return exports.export_session(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail='session not found') from exc

    @app.get('/v1/hands/{hand_id}/spots/normalized')
    def normalized_spot(hand_id: str) -> dict:
        try:
            return solver_labels.normalized_spot_for_hand(hand_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail='hand not found') from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post('/v1/spots/packs/{spot_id}/compare-solver-like')
    def compare_solver_like(spot_id: str) -> dict:
        try:
            return solver_labels.compare_spot_pack(spot_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail='spot pack not found') from exc

    @app.get('/v1/solver/catalog')
    def solver_catalog() -> dict:
        return {'labeler': 'solver_like_v1_rule', 'notes': 'offline heuristic labeler until true solver integration'}

    @app.post('/v1/models/train/policy-table')
    def train_policy_table() -> dict:
        return models.train_policy_table()

    @app.get('/v1/models')
    def list_models() -> dict:
        return {'models': models.list_models()}

    @app.post('/v1/models/{model_id}/evaluate')
    def evaluate_model(model_id: str) -> dict:
        try:
            return models.evaluate_model_on_spot_packs(model_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail='model not found') from exc

    @app.post('/v1/datasets/build')
    def build_dataset(dataset_name: str = 'master_v1') -> dict:
        return datasets.build_master_dataset(dataset_name=dataset_name)

    @app.get('/v1/datasets')
    def list_datasets() -> dict:
        return {'datasets': datasets.list_datasets()}

    @app.post('/v1/models/{model_id}/evaluate-dataset/{dataset_id}')
    def evaluate_model_dataset(model_id: str, dataset_id: str, split: str | None = None) -> dict:
        try:
            return evaluation.evaluate_model(model_id, dataset_id, split=split)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail='dataset not found') from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail='model not found') from exc

    @app.get('/v1/external-solver/catalog')
    def external_solver_catalog() -> dict:
        return external_solver.catalog()

    @app.post('/v1/external-solver/{solver_name}/compare/{spot_id}')
    def compare_external_solver(solver_name: str, spot_id: str) -> dict:
        try:
            return external_solver.compare_spot_pack(solver_name, spot_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail='solver or spot pack not found') from exc

    @app.post('/v1/tournaments/round-robin')
    def round_robin(payload: TournamentRequest) -> dict:
        try:
            return tournaments.round_robin(payload.entrants, hands_per_match=payload.hands_per_match, seed_base=payload.seed_base, stacks=payload.stacks)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f'entrant not found: {exc}') from exc

    @app.get('/v1/sessions/{session_id}/curriculum')
    def get_curriculum(session_id: str) -> dict:
        try:
            return curriculum.build_for_session(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail='session not found') from exc

    @app.get('/v1/system/readiness')
    def system_readiness() -> dict:
        return readiness.snapshot()

    @app.get('/v1/sessions/{session_id}/opponent-profile')
    def get_opponent_profile(session_id: str) -> dict:
        try:
            return opponent_profiles.session_profiles(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail='session not found') from exc

    @app.post('/v1/benchmark/adaptive')
    def run_adaptive_benchmark(payload: AdaptiveBenchmarkRequest) -> dict:
        return opponent_profiles.adaptive_benchmark(num_hands=payload.num_hands, stacks=payload.stacks, seed_base=payload.seed_base)

    @app.get('/v1/sessions/{session_id}/coach-report')
    def get_coach_report(session_id: str) -> dict:
        try:
            return coach.session_report(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail='session not found') from exc

    @app.get('/v1/hands/{hand_id}/review')
    def review_hand(hand_id: str) -> dict:
        try:
            return coach.hand_review(hand_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail='hand not found') from exc


    @app.post('/v1/governance/model-cards/{model_id}')
    def build_model_card(model_id: str) -> dict:
        try:
            return governance.build_model_card(model_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail='model not found') from exc

    @app.get('/v1/governance/model-cards')
    def list_model_cards() -> dict:
        return {'model_cards': governance.list_model_cards()}

    @app.post('/v1/release/gate/{model_id}/{dataset_id}')
    def evaluate_release_gate(model_id: str, dataset_id: str, payload: ReleaseGateRequest) -> dict:
        try:
            return release_gate.evaluate_candidate(model_id, dataset_id, min_accuracy=payload.min_accuracy, min_rows=payload.min_rows)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail='dataset not found') from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail='model not found') from exc

    @app.post('/v1/hands/batch-generate')
    def generate_batch(payload: BatchGenerateRequest) -> dict:
        return batch_generation.generate(batch_name=payload.batch_name, num_hands=payload.num_hands, seed_base=payload.seed_base, stacks=payload.stacks)

    @app.post('/v1/regression/suites/build')
    def build_regression_suite(payload: RegressionSuiteRequest) -> dict:
        return regression_suites.build(suite_name=payload.suite_name)

    @app.post('/v1/models/{model_id}/calibrate/{dataset_id}')
    def calibrate_model(model_id: str, dataset_id: str, split: str | None = None) -> dict:
        try:
            return calibration.calibrate(model_id, dataset_id, split=split)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail='dataset not found') from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail='model not found') from exc

    @app.get('/v1/deploy/manifest')
    def deploy_manifest() -> dict:
        return deploy.manifest()

    @app.get('/v1/release/notes')
    def get_release_notes() -> dict:
        return release_notes.current_notes()

    @app.get('/v1/system/alpha-candidate')
    def get_alpha_candidate(model_id: str, dataset_id: str) -> dict:
        try:
            return alpha_candidate.snapshot(model_id, dataset_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail='dataset not found') from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail='model not found') from exc

    app.state.engine = engine
    app.state.store = store
    app.state.runtimes = runtimes
    return app


app = create_app()
