from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from services.experiment_runner.service import ExperimentConfig
from services.session_service.service import SessionConfig
from services.cqrs.saga import InProcessSaga, SagaStep


@dataclass(frozen=True)
class SessionH2HCommand:
    num_hands: int
    stacks: tuple[int, int]
    seed_base: int
    session_name: str


@dataclass(frozen=True)
class SmokeBenchmarkCommand:
    num_hands: int
    stacks: tuple[int, int]
    seed_base: int


@dataclass(frozen=True)
class AdaptiveBenchmarkCommand:
    num_hands: int
    stacks: tuple[int, int]
    seed_base: int


@dataclass(frozen=True)
class BenchmarkH2HCommand:
    num_matches: int
    hands_per_match: int
    stacks: tuple[int, int]
    seed_base: int


@dataclass(frozen=True)
class RoundRobinTournamentCommand:
    entrants: list[str]
    hands_per_match: int
    seed_base: int
    stacks: tuple[int, int]


class CommandBus:
    """Internal command dispatcher (CQRS write side)."""

    def __init__(
        self,
        *,
        sessions: Any,
        benchmark: Any,
        experiments: Any,
        opponent_profiles: Any,
        tournaments: Any,
    ) -> None:
        self._sessions = sessions
        self._benchmark = benchmark
        self._experiments = experiments
        self._opponent_profiles = opponent_profiles
        self._tournaments = tournaments

    def _run_compound(self, steps: list[SagaStep[Any]]) -> Any:
        saga = InProcessSaga()
        results = saga.run(steps)
        return results[-1] if results else None

    def run_session_h2h(self, command: SessionH2HCommand) -> dict[str, Any]:
        return cast(
            dict[str, Any],
            self._run_compound(
                [
                    SagaStep(
                        name="session.h2h.run",
                        action=lambda: self._sessions.run_h2h(
                            SessionConfig(
                                num_hands=command.num_hands,
                                stacks=command.stacks,
                                seed_base=command.seed_base,
                                session_name=command.session_name,
                            )
                        ),
                    )
                ]
            ),
        )

    def run_smoke_benchmark(self, command: SmokeBenchmarkCommand) -> dict[str, Any]:
        return cast(
            dict[str, Any],
            self._run_compound(
                [
                    SagaStep(
                        name="benchmark.smoke.run",
                        action=lambda: self._benchmark.smoke(
                            num_hands=command.num_hands,
                            stacks=command.stacks,
                            seed_base=command.seed_base,
                        ),
                    )
                ]
            ),
        )

    def run_adaptive_benchmark(
        self, command: AdaptiveBenchmarkCommand
    ) -> dict[str, Any]:
        return cast(
            dict[str, Any],
            self._run_compound(
                [
                    SagaStep(
                        name="benchmark.adaptive.run",
                        action=lambda: self._opponent_profiles.adaptive_benchmark(
                            num_hands=command.num_hands,
                            stacks=command.stacks,
                            seed_base=command.seed_base,
                        ),
                    )
                ]
            ),
        )

    def run_h2h_benchmark(self, command: BenchmarkH2HCommand) -> dict[str, Any]:
        return cast(
            dict[str, Any],
            self._run_compound(
                [
                    SagaStep(
                        name="benchmark.h2h.run",
                        action=lambda: self._benchmark.h2h_eval(
                            num_matches=command.num_matches,
                            hands_per_match=command.hands_per_match,
                            stacks=command.stacks,
                            seed_base=command.seed_base,
                        ),
                    )
                ]
            ),
        )

    def run_experiment_h2h(self, command: BenchmarkH2HCommand) -> dict[str, Any]:
        return cast(
            dict[str, Any],
            self._run_compound(
                [
                    SagaStep(
                        name="experiment.h2h.run",
                        action=lambda: self._experiments.run_h2h(
                            ExperimentConfig(
                                num_matches=command.num_matches,
                                hands_per_match=command.hands_per_match,
                                stacks=command.stacks,
                                seed_base=command.seed_base,
                            )
                        ),
                    )
                ]
            ),
        )

    def run_round_robin(self, command: RoundRobinTournamentCommand) -> dict[str, Any]:
        return cast(
            dict[str, Any],
            self._run_compound(
                [
                    SagaStep(
                        name="tournament.round_robin.run",
                        action=lambda: self._tournaments.round_robin(
                            command.entrants,
                            hands_per_match=command.hands_per_match,
                            seed_base=command.seed_base,
                            stacks=command.stacks,
                        ),
                    )
                ]
            ),
        )


class QueryBus:
    """Internal query dispatcher (CQRS read side)."""

    def __init__(
        self,
        *,
        store: Any,
        analytics: Any,
        readiness: Any,
        alpha_candidate: Any,
        models: Any,
        datasets: Any,
    ) -> None:
        self._store = store
        self._analytics = analytics
        self._readiness = readiness
        self._alpha_candidate = alpha_candidate
        self._models = models
        self._datasets = datasets

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        return cast(dict[str, Any] | None, self._store.get_session(session_id))

    def get_session_traces(self, session_id: str) -> list[dict[str, Any]]:
        return cast(
            list[dict[str, Any]],
            self._store.get_decision_traces(session_id=session_id),
        )

    def get_session_analytics(self, session_id: str) -> dict[str, Any]:
        return cast(dict[str, Any], self._analytics.session_analytics(session_id))

    def get_readiness(self) -> dict[str, Any]:
        return cast(dict[str, Any], self._readiness.snapshot())

    def get_alpha_candidate(self, model_id: str, dataset_id: str) -> dict[str, Any]:
        return cast(
            dict[str, Any], self._alpha_candidate.snapshot(model_id, dataset_id)
        )

    def list_models(self) -> dict[str, Any]:
        return cast(dict[str, Any], self._models.list_models())

    def list_datasets(self) -> dict[str, Any]:
        return cast(dict[str, Any], self._datasets.list_datasets())
