
from __future__ import annotations

from dataclasses import dataclass

from packages.baseline_agent import BaselineAgent
from packages.engine import GameEngine


@dataclass
class MatchSummary:
    hands_played: int
    seat0_profit: int
    seat1_profit: int
    seat0_wins: int
    seat1_wins: int
    draws: int
    avg_actions_per_hand: float
    bb_per_100_seat0: float
    bb_per_100_seat1: float
    seat0_name: str
    seat1_name: str


class MatchHarness:
    def __init__(self, engine: GameEngine | None = None, seat0_agent=None, seat1_agent=None) -> None:
        self.engine = engine or GameEngine()
        self.seat0_agent = seat0_agent or BaselineAgent()
        self.seat1_agent = seat1_agent or BaselineAgent()

    def run(self, *, num_hands: int = 100, stacks: tuple[int, int] = (100, 100), seed_base: int = 10_000) -> dict:
        seat0_profit = 0
        seat1_profit = 0
        seat0_wins = 0
        seat1_wins = 0
        draws = 0
        total_actions = 0

        for index in range(num_hands):
            button = index % 2
            runtime = self.engine.start_new_hand(stacks=stacks, button_seat=button, seed=seed_base + index)
            while not runtime.state.is_terminal:
                seat = runtime.state.acting_seat
                agent = self.seat0_agent if seat == 0 else self.seat1_agent
                decision = agent.decide(runtime, self.engine)
                self.engine.apply_action(runtime, decision.action_type, decision.amount)
            end0 = runtime.state.players[0].stack
            end1 = runtime.state.players[1].stack
            seat0_profit += end0 - stacks[0]
            seat1_profit += end1 - stacks[1]
            total_actions += len(runtime.state.actions)
            if runtime.state.winner_seat == 0:
                seat0_wins += 1
            elif runtime.state.winner_seat == 1:
                seat1_wins += 1
            else:
                draws += 1

        big_blind = self.engine.big_blind
        summary = MatchSummary(
            hands_played=num_hands,
            seat0_profit=seat0_profit,
            seat1_profit=seat1_profit,
            seat0_wins=seat0_wins,
            seat1_wins=seat1_wins,
            draws=draws,
            avg_actions_per_hand=total_actions / num_hands if num_hands else 0.0,
            bb_per_100_seat0=(seat0_profit / big_blind) * (100 / num_hands) if num_hands else 0.0,
            bb_per_100_seat1=(seat1_profit / big_blind) * (100 / num_hands) if num_hands else 0.0,
            seat0_name=self.seat0_agent.__class__.__name__,
            seat1_name=self.seat1_agent.__class__.__name__,
        )
        return summary.__dict__
