from __future__ import annotations

import os
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import repeat
from pathlib import Path
from statistics import mean
from typing import Any
from uuid import uuid4

from packages.baseline_agent.agent import BaselineAgent
from packages.common.types import ActionType
from packages.engine.engine import GameEngine
from packages.persistence.sqlite_store import SqliteHandStore


@dataclass
class SessionConfig:
    num_hands: int = 100
    stacks: tuple[int, int] = (100, 100)
    seed_base: int = 30_000
    session_name: str = "local_h2h_session"


@dataclass
class HandSimulationResult:
    index: int
    seed: int
    button: int
    hand_id: str
    initial_snapshot: dict[str, Any]
    actions: list[tuple[int, str, int, int]]
    snapshots: list[tuple[int, dict[str, Any], str]]
    traces: list[tuple[int, dict[str, Any]]]
    end_stacks: tuple[int, int]
    winner_seat: int | None
    total_actions: int


_WORKER_ENGINE: GameEngine | None = None
_WORKER_AGENTS: tuple[BaselineAgent, BaselineAgent] | None = None


def _worker_context() -> tuple[GameEngine, BaselineAgent, BaselineAgent]:
    global _WORKER_ENGINE
    global _WORKER_AGENTS
    if _WORKER_ENGINE is None or _WORKER_AGENTS is None:
        _WORKER_ENGINE = GameEngine()
        _WORKER_AGENTS = (BaselineAgent(), BaselineAgent())
    return _WORKER_ENGINE, _WORKER_AGENTS[0], _WORKER_AGENTS[1]


def _simulate_hand(
    index: int, stacks: tuple[int, int], seed_base: int
) -> HandSimulationResult:
    engine, seat0_agent, seat1_agent = _worker_context()
    button = index % 2
    seed = seed_base + index
    runtime = engine.start_new_hand(
        stacks=stacks,
        button_seat=button,
        seed=seed,
    )
    initial_snapshot = engine.state_snapshot(runtime)
    action_order = 0
    action_batch: list[tuple[int, str, int, int]] = []
    trace_batch: list[tuple[int, dict[str, Any]]] = []
    while not runtime.state.is_terminal:
        seat = runtime.state.acting_seat
        agent = seat0_agent if seat == 0 else seat1_agent
        decision = agent.decide(runtime, engine)
        if decision.trace is not None:
            trace_payload = decision.trace.model_dump(mode="json")
            trace_batch.append((int(trace_payload["actor_seat"]), trace_payload))
        engine.apply_action(runtime, decision.action_type, decision.amount)
        action_batch.append(
            (
                int(seat) if seat is not None else -1,
                decision.action_type.value,
                decision.amount,
                action_order,
            )
        )
        action_order += 1
    snapshot_batch = [(1, engine.state_snapshot(runtime), "session_terminal")]
    return HandSimulationResult(
        index=index,
        seed=seed,
        button=button,
        hand_id=runtime.state.hand_id,
        initial_snapshot=initial_snapshot,
        actions=action_batch,
        snapshots=snapshot_batch,
        traces=trace_batch,
        end_stacks=(runtime.state.players[0].stack, runtime.state.players[1].stack),
        winner_seat=runtime.state.winner_seat,
        total_actions=len(runtime.state.actions),
    )


class SessionRunner:
    def __init__(
        self,
        engine: GameEngine,
        store: SqliteHandStore,
        report_dir: str = "var/reports",
    ) -> None:
        self.engine = engine
        self.store = store
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def run_h2h(self, config: SessionConfig) -> dict[str, Any]:
        session_id = str(uuid4())
        self.store.create_session(
            session_id=session_id, session_type="h2h_local", config=asdict(config)
        )
        try:
            seat0_profit = 0
            seat1_profit = 0
            seat0_wins = 0
            seat1_wins = 0
            draws = 0
            total_actions = 0
            traces: list[dict[str, Any]] = []
            hand_ids: list[str] = []
            detected_cpus = os.cpu_count() or 1
            max_workers = max(
                1,
                min(
                    detected_cpus,
                    int(
                        os.getenv(
                            "POKER_SESSION_MAX_WORKERS", str(min(4, detected_cpus))
                        )
                    ),
                ),
            )
            parallel_threshold = int(
                os.getenv("POKER_SESSION_PARALLEL_THRESHOLD", "9999")
            )
            if config.num_hands >= parallel_threshold and max_workers > 1:
                with ProcessPoolExecutor(max_workers=max_workers) as pool:
                    hand_results = list(
                        pool.map(
                            _simulate_hand,
                            range(config.num_hands),
                            repeat(config.stacks),
                            repeat(config.seed_base),
                            chunksize=4,
                        )
                    )
            else:
                hand_results = [
                    _simulate_hand(index, config.stacks, config.seed_base)
                    for index in range(config.num_hands)
                ]

            for result in sorted(hand_results, key=lambda item: item.index):
                hand_ids.append(result.hand_id)
                with self.store.uow() as uow:
                    uow.create_hand(
                        hand_id=result.hand_id,
                        session_id=session_id,
                        button_seat=result.button,
                        stacks=config.stacks,
                        seed=result.seed,
                        deck_prefix=[],
                        initial_snapshot=result.initial_snapshot,
                    )
                    uow.append_hand_batch(
                        hand_id=result.hand_id,
                        session_id=session_id,
                        actions=result.actions,
                        snapshots=result.snapshots,
                        decision_traces=result.traces,
                    )
                traces.extend(trace for _, trace in result.traces)
                end0, end1 = result.end_stacks
                seat0_profit += end0 - config.stacks[0]
                seat1_profit += end1 - config.stacks[1]
                total_actions += result.total_actions
                if result.winner_seat == 0:
                    seat0_wins += 1
                elif result.winner_seat == 1:
                    seat1_wins += 1
                else:
                    draws += 1

            summary = self._build_summary(
                config=config,
                seat0_profit=seat0_profit,
                seat1_profit=seat1_profit,
                seat0_wins=seat0_wins,
                seat1_wins=seat1_wins,
                draws=draws,
                total_actions=total_actions,
                traces=traces,
                hand_ids=hand_ids,
            )
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            report_path = self.report_dir / f"session_{stamp}.json"
            report_path.write_text(json.dumps(summary, indent=2))
            summary["report_path"] = str(report_path)
            self.store.update_session_summary(session_id=session_id, summary=summary)
            payload = {"session_id": session_id, **summary}
            return payload
        except Exception:
            # SAGA compensation: remove partial session artifacts on failure.
            self.store.delete_session_data(session_id=session_id)
            raise

    def _build_summary(
        self,
        *,
        config: SessionConfig,
        seat0_profit: int,
        seat1_profit: int,
        seat0_wins: int,
        seat1_wins: int,
        draws: int,
        total_actions: int,
        traces: list[dict[str, Any]],
        hand_ids: list[str],
    ) -> dict[str, Any]:
        big_blind = self.engine.big_blind
        priced = [t for t in traces if int(t.get("to_call", 0)) > 0]
        continue_traces = [
            t
            for t in priced
            if t.get("action_type")
            in {ActionType.CALL.value, ActionType.RAISE.value, ActionType.ALL_IN.value}
        ]

        def _edge(trace: dict[str, Any]) -> float:
            return float(trace["estimated_equity"]) - float(trace["pot_odds"])

        def _continue_ev_proxy(trace: dict[str, Any]) -> float:
            pot = float(trace["pot"])
            to_call = float(trace["to_call"])
            equity = float(trace["estimated_equity"])
            return (equity * (pot + to_call)) - to_call

        seat_metrics: dict[str, dict[str, Any]] = {}
        for seat in (0, 1):
            seat_priced = [t for t in priced if int(t["actor_seat"]) == seat]
            seat_continue = [t for t in continue_traces if int(t["actor_seat"]) == seat]
            seat_metrics[f"seat{seat}"] = {
                "trace_count": len([t for t in traces if int(t["actor_seat"]) == seat]),
                "priced_decision_count": len(seat_priced),
                "continue_decision_count": len(seat_continue),
                "avg_priced_edge": (
                    round(mean(_edge(t) for t in seat_priced), 4)
                    if seat_priced
                    else 0.0
                ),
                "continue_ev_proxy_sum": round(
                    sum(_continue_ev_proxy(t) for t in seat_continue), 4
                ),
                "positive_edge_continue_rate": (
                    round(
                        sum(1 for t in seat_continue if _edge(t) >= 0)
                        / len(seat_continue),
                        4,
                    )
                    if seat_continue
                    else 0.0
                ),
            }

        return {
            "session_name": config.session_name,
            "session_type": "h2h_local",
            "config": asdict(config),
            "hands_played": config.num_hands,
            "hand_ids": hand_ids,
            "seat0_profit": seat0_profit,
            "seat1_profit": seat1_profit,
            "seat0_wins": seat0_wins,
            "seat1_wins": seat1_wins,
            "draws": draws,
            "avg_actions_per_hand": (
                round(total_actions / config.num_hands, 4) if config.num_hands else 0.0
            ),
            "bb_per_100_seat0": (
                round((seat0_profit / big_blind) * (100 / config.num_hands), 4)
                if config.num_hands
                else 0.0
            ),
            "bb_per_100_seat1": (
                round((seat1_profit / big_blind) * (100 / config.num_hands), 4)
                if config.num_hands
                else 0.0
            ),
            "decision_trace_count": len(traces),
            "priced_decision_count": len(priced),
            "continue_decision_count": len(continue_traces),
            "avg_priced_edge": (
                round(mean(_edge(t) for t in priced), 4) if priced else 0.0
            ),
            "continue_ev_proxy_sum": round(
                sum(_continue_ev_proxy(t) for t in continue_traces), 4
            ),
            "positive_edge_continue_rate": (
                round(
                    sum(1 for t in continue_traces if _edge(t) >= 0)
                    / len(continue_traces),
                    4,
                )
                if continue_traces
                else 0.0
            ),
            "evaluation_note": "continue_ev_proxy_sum and avg_priced_edge are local proxy metrics derived from decision traces; they are not true game-theoretic EV.",
            "seat_metrics": seat_metrics,
        }
