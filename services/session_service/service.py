
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from uuid import uuid4

from packages.baseline_agent import BaselineAgent
from packages.common.types import ActionType
from packages.engine import GameEngine
from packages.persistence import SqliteHandStore


@dataclass
class SessionConfig:
    num_hands: int = 100
    stacks: tuple[int, int] = (100, 100)
    seed_base: int = 30_000
    session_name: str = 'local_h2h_session'


class SessionRunner:
    def __init__(self, engine: GameEngine, store: SqliteHandStore, report_dir: str = 'var/reports') -> None:
        self.engine = engine
        self.store = store
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def run_h2h(self, config: SessionConfig) -> dict:
        session_id = str(uuid4())
        self.store.create_session(session_id=session_id, session_type='h2h_local', config=asdict(config))

        seat0_agent = BaselineAgent()
        seat1_agent = BaselineAgent()

        seat0_profit = 0
        seat1_profit = 0
        seat0_wins = 0
        seat1_wins = 0
        draws = 0
        total_actions = 0
        traces: list[dict] = []
        hand_ids: list[str] = []

        for index in range(config.num_hands):
            button = index % 2
            runtime = self.engine.start_new_hand(
                stacks=config.stacks,
                button_seat=button,
                seed=config.seed_base + index,
            )
            hand_ids.append(runtime.state.hand_id)
            self.store.create_hand(
                hand_id=runtime.state.hand_id,
                session_id=session_id,
                button_seat=button,
                stacks=config.stacks,
                seed=config.seed_base + index,
                deck_prefix=[],
                initial_snapshot=self.engine.state_snapshot(runtime),
            )
            while not runtime.state.is_terminal:
                seat = runtime.state.acting_seat
                agent = seat0_agent if seat == 0 else seat1_agent
                decision = agent.decide(runtime, self.engine)
                if decision.trace is not None:
                    trace_payload = decision.trace.model_dump(mode='json')
                    traces.append(trace_payload)
                    self.store.append_decision_trace(
                        session_id=session_id,
                        hand_id=runtime.state.hand_id,
                        actor_seat=trace_payload['actor_seat'],
                        trace=trace_payload,
                    )
                self.engine.apply_action(runtime, decision.action_type, decision.amount)
                self.store.append_action(
                    hand_id=runtime.state.hand_id,
                    actor_seat=int(seat) if seat is not None else -1,
                    action_type=decision.action_type.value,
                    amount=decision.amount,
                )
                self.store.append_snapshot(
                    hand_id=runtime.state.hand_id,
                    snapshot=self.engine.state_snapshot(runtime),
                    label='session_progress',
                )
            end0 = runtime.state.players[0].stack
            end1 = runtime.state.players[1].stack
            seat0_profit += end0 - config.stacks[0]
            seat1_profit += end1 - config.stacks[1]
            total_actions += len(runtime.state.actions)
            if runtime.state.winner_seat == 0:
                seat0_wins += 1
            elif runtime.state.winner_seat == 1:
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
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        report_path = self.report_dir / f'session_{stamp}.json'
        report_path.write_text(json.dumps(summary, indent=2))
        summary['report_path'] = str(report_path)
        self.store.update_session_summary(session_id=session_id, summary=summary)
        payload = {'session_id': session_id, **summary}
        return payload

    def _build_summary(self, *, config: SessionConfig, seat0_profit: int, seat1_profit: int, seat0_wins: int, seat1_wins: int, draws: int, total_actions: int, traces: list[dict], hand_ids: list[str]) -> dict:
        big_blind = self.engine.big_blind
        priced = [t for t in traces if int(t.get('to_call', 0)) > 0]
        continue_traces = [
            t for t in priced if t.get('action_type') in {ActionType.CALL.value, ActionType.RAISE.value, ActionType.ALL_IN.value}
        ]

        def _edge(trace: dict) -> float:
            return float(trace['estimated_equity']) - float(trace['pot_odds'])

        def _continue_ev_proxy(trace: dict) -> float:
            pot = float(trace['pot'])
            to_call = float(trace['to_call'])
            equity = float(trace['estimated_equity'])
            return (equity * (pot + to_call)) - to_call

        seat_metrics: dict[str, dict] = {}
        for seat in (0, 1):
            seat_priced = [t for t in priced if int(t['actor_seat']) == seat]
            seat_continue = [t for t in continue_traces if int(t['actor_seat']) == seat]
            seat_metrics[f'seat{seat}'] = {
                'trace_count': len([t for t in traces if int(t['actor_seat']) == seat]),
                'priced_decision_count': len(seat_priced),
                'continue_decision_count': len(seat_continue),
                'avg_priced_edge': round(mean(_edge(t) for t in seat_priced), 4) if seat_priced else 0.0,
                'continue_ev_proxy_sum': round(sum(_continue_ev_proxy(t) for t in seat_continue), 4),
                'positive_edge_continue_rate': round(sum(1 for t in seat_continue if _edge(t) >= 0) / len(seat_continue), 4) if seat_continue else 0.0,
            }

        return {
            'session_name': config.session_name,
            'session_type': 'h2h_local',
            'config': asdict(config),
            'hands_played': config.num_hands,
            'hand_ids': hand_ids,
            'seat0_profit': seat0_profit,
            'seat1_profit': seat1_profit,
            'seat0_wins': seat0_wins,
            'seat1_wins': seat1_wins,
            'draws': draws,
            'avg_actions_per_hand': round(total_actions / config.num_hands, 4) if config.num_hands else 0.0,
            'bb_per_100_seat0': round((seat0_profit / big_blind) * (100 / config.num_hands), 4) if config.num_hands else 0.0,
            'bb_per_100_seat1': round((seat1_profit / big_blind) * (100 / config.num_hands), 4) if config.num_hands else 0.0,
            'decision_trace_count': len(traces),
            'priced_decision_count': len(priced),
            'continue_decision_count': len(continue_traces),
            'avg_priced_edge': round(mean(_edge(t) for t in priced), 4) if priced else 0.0,
            'continue_ev_proxy_sum': round(sum(_continue_ev_proxy(t) for t in continue_traces), 4),
            'positive_edge_continue_rate': round(sum(1 for t in continue_traces if _edge(t) >= 0) / len(continue_traces), 4) if continue_traces else 0.0,
            'evaluation_note': 'continue_ev_proxy_sum and avg_priced_edge are local proxy metrics derived from decision traces; they are not true game-theoretic EV.',
            'seat_metrics': seat_metrics,
        }
