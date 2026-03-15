from __future__ import annotations

import os
from dataclasses import dataclass

from packages.baseline_agent import BaselineAgent
from packages.engine import GameEngine, HandRuntime
from packages.persistence import SqliteHandStore
from services.alpha_candidate_service import AlphaCandidateService
from services.analytics_service import SessionAnalyticsService
from services.batch_generation_service import BatchGenerationService
from services.benchmark_service import BenchmarkService
from services.calibration_service import CalibrationService
from services.coach_service import CoachService
from services.cqrs import CommandBus, QueryBus
from services.curriculum_service import CurriculumService
from services.dataset_service import DatasetService
from services.deploy_service import DeployService
from services.evaluation_service import EvaluationService
from services.experiment_runner import ExperimentRunner
from services.export_service import ExportService
from services.external_solver_service import ExternalSolverService
from services.governance_service import GovernanceService
from services.model_service import ModelService
from services.opponent_profile_service import OpponentProfileService
from services.readiness_service import ReadinessService
from services.regression_suite_service import RegressionSuiteService
from services.release_gate_service import ReleaseGateService
from services.release_notes_service import ReleaseNotesService
from services.replay_service.service import ReplayService
from services.session_service import SessionRunner
from services.solver_label_service import SolverLabelService
from services.spot_pack_service import SpotPackService
from services.taxonomy_service import TaxonomyService
from services.tournament_service import TournamentService


@dataclass
class AppContainer:
    engine: GameEngine
    agent: BaselineAgent
    store: SqliteHandStore
    replay: ReplayService
    benchmark: BenchmarkService
    experiments: ExperimentRunner
    sessions: SessionRunner
    spot_packs: SpotPackService
    taxonomy: TaxonomyService
    analytics: SessionAnalyticsService
    exports: ExportService
    solver_labels: SolverLabelService
    models: ModelService
    opponent_profiles: OpponentProfileService
    coach: CoachService
    datasets: DatasetService
    evaluation: EvaluationService
    external_solver: ExternalSolverService
    tournaments: TournamentService
    curriculum: CurriculumService
    readiness: ReadinessService
    governance: GovernanceService
    release_gate: ReleaseGateService
    batch_generation: BatchGenerationService
    regression_suites: RegressionSuiteService
    calibration: CalibrationService
    deploy: DeployService
    release_notes: ReleaseNotesService
    alpha_candidate: AlphaCandidateService
    command_bus: CommandBus
    query_bus: QueryBus
    runtimes: dict[str, HandRuntime]


def build_container(db_path: str | None = None, base_dir: str = ".") -> AppContainer:
    engine = GameEngine()
    agent = BaselineAgent()
    store = SqliteHandStore(
        db_path or os.getenv("POKER_AI_DB", "var/poker_ai_local.db")
    )
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
    readiness = ReadinessService(base_dir=base_dir)
    governance = GovernanceService()
    release_gate = ReleaseGateService(readiness=readiness, evaluation=evaluation)
    batch_generation = BatchGenerationService(session_runner=sessions)
    regression_suites = RegressionSuiteService()
    calibration = CalibrationService(evaluation=evaluation)
    deploy = DeployService(base_dir=base_dir)
    release_notes = ReleaseNotesService(docs_dir="docs")
    alpha_candidate = AlphaCandidateService(
        release_gate=release_gate,
        governance=governance,
        deploy=deploy,
    )
    command_bus = CommandBus(
        sessions=sessions,
        benchmark=benchmark,
        experiments=experiments,
        opponent_profiles=opponent_profiles,
        tournaments=tournaments,
    )
    query_bus = QueryBus(
        store=store,
        analytics=analytics,
        readiness=readiness,
        alpha_candidate=alpha_candidate,
        models=models,
        datasets=datasets,
    )

    return AppContainer(
        engine=engine,
        agent=agent,
        store=store,
        replay=replay,
        benchmark=benchmark,
        experiments=experiments,
        sessions=sessions,
        spot_packs=spot_packs,
        taxonomy=taxonomy,
        analytics=analytics,
        exports=exports,
        solver_labels=solver_labels,
        models=models,
        opponent_profiles=opponent_profiles,
        coach=coach,
        datasets=datasets,
        evaluation=evaluation,
        external_solver=external_solver,
        tournaments=tournaments,
        curriculum=curriculum,
        readiness=readiness,
        governance=governance,
        release_gate=release_gate,
        batch_generation=batch_generation,
        regression_suites=regression_suites,
        calibration=calibration,
        deploy=deploy,
        release_notes=release_notes,
        alpha_candidate=alpha_candidate,
        command_bus=command_bus,
        query_bus=query_bus,
        runtimes={},
    )
