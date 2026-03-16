from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from apps.api.contracts import (
    AdaptiveBenchmarkRequest,
    H2HBenchmarkRequest,
    SmokeBenchmarkRequest,
    TournamentRequest,
)
from apps.api.routers.utils import get_container
from packages.auth.dependencies import require_write
from packages.auth.models import User
from services.cqrs import (
    AdaptiveBenchmarkCommand,
    BenchmarkH2HCommand,
    RoundRobinTournamentCommand,
    SmokeBenchmarkCommand,
)

router = APIRouter(prefix="/v1")


@router.post("/benchmark/smoke")
def smoke_benchmark(payload: SmokeBenchmarkRequest, request: Request, user: User = Depends(require_write)) -> dict:
    container = get_container(request)
    container.audit.log_access(user.username, "/v1/benchmark/smoke", "run")
    return container.command_bus.run_smoke_benchmark(
        SmokeBenchmarkCommand(
            num_hands=payload.num_hands,
            stacks=payload.stacks,
            seed_base=payload.seed_base,
        )
    )


@router.post("/benchmark/h2h")
def h2h_benchmark(payload: H2HBenchmarkRequest, request: Request, user: User = Depends(require_write)) -> dict:
    container = get_container(request)
    container.audit.log_access(user.username, "/v1/benchmark/h2h", "run")
    return container.command_bus.run_h2h_benchmark(
        BenchmarkH2HCommand(
            num_matches=payload.num_matches,
            hands_per_match=payload.hands_per_match,
            stacks=payload.stacks,
            seed_base=payload.seed_base,
        )
    )


@router.post("/experiments/run")
def run_experiment(payload: H2HBenchmarkRequest, request: Request, user: User = Depends(require_write)) -> dict:
    container = get_container(request)
    return container.command_bus.run_experiment_h2h(
        BenchmarkH2HCommand(
            num_matches=payload.num_matches,
            hands_per_match=payload.hands_per_match,
            stacks=payload.stacks,
            seed_base=payload.seed_base,
        )
    )


@router.post("/benchmark/adaptive")
def run_adaptive_benchmark(payload: AdaptiveBenchmarkRequest, request: Request, user: User = Depends(require_write)) -> dict:
    container = get_container(request)
    return container.command_bus.run_adaptive_benchmark(
        AdaptiveBenchmarkCommand(
            num_hands=payload.num_hands,
            stacks=payload.stacks,
            seed_base=payload.seed_base,
        )
    )


@router.post("/tournaments/round-robin")
def round_robin(payload: TournamentRequest, request: Request, user: User = Depends(require_write)) -> dict:
    container = get_container(request)
    try:
        return container.command_bus.run_round_robin(
            RoundRobinTournamentCommand(
                entrants=payload.entrants,
                hands_per_match=payload.hands_per_match,
                seed_base=payload.seed_base,
                stacks=payload.stacks,
            )
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail=f"entrant not found: {exc}"
        ) from exc
