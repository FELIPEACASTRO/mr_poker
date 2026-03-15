from __future__ import annotations

from collections import defaultdict
from typing import Any

from packages.adaptive_agent import AdaptiveBaselineAgent
from packages.baseline_agent import BaselineAgent
from packages.engine import GameEngine
from packages.harness import MatchHarness
from packages.policy_agent import PolicyTableAgent


class TournamentService:
    def __init__(self, engine: GameEngine, model_registry) -> None:
        self.engine = engine
        self.model_registry = model_registry

    def _agent_factory(self, name: str):
        if name == 'baseline':
            return BaselineAgent()
        if name == 'adaptive':
            return AdaptiveBaselineAgent()
        if name.startswith('policy:'):
            return PolicyTableAgent(self.model_registry.load(name.split(':', 1)[1]))
        raise KeyError(name)

    def round_robin(self, entrants: list[str], *, hands_per_match: int = 30, seed_base: int = 70_000, stacks: tuple[int, int] = (100, 100)) -> dict[str, Any]:
        leaderboard: dict[str, dict[str, Any]] = defaultdict(lambda: {'matches': 0, 'hands': 0, 'profit': 0, 'wins': 0})
        fixtures = []
        seed = seed_base
        for i, seat0_name in enumerate(entrants):
            for seat1_name in entrants[i + 1:]:
                harness = MatchHarness(engine=self.engine, seat0_agent=self._agent_factory(seat0_name), seat1_agent=self._agent_factory(seat1_name))
                result = harness.run(num_hands=hands_per_match, stacks=stacks, seed_base=seed)
                seed += hands_per_match + 17
                fixtures.append({'seat0': seat0_name, 'seat1': seat1_name, 'result': result})
                leaderboard[seat0_name]['matches'] += 1
                leaderboard[seat1_name]['matches'] += 1
                leaderboard[seat0_name]['hands'] += hands_per_match
                leaderboard[seat1_name]['hands'] += hands_per_match
                leaderboard[seat0_name]['profit'] += result['seat0_profit']
                leaderboard[seat1_name]['profit'] += result['seat1_profit']
                leaderboard[seat0_name]['wins'] += result['seat0_wins']
                leaderboard[seat1_name]['wins'] += result['seat1_wins']
        rows = []
        for entrant, stats in leaderboard.items():
            hands = stats['hands']
            rows.append({'entrant': entrant, 'matches': stats['matches'], 'hands': hands, 'profit': stats['profit'], 'wins': stats['wins'], 'bb_per_100': round((stats['profit']/self.engine.big_blind) * (100/hands), 4) if hands else 0.0})
        rows.sort(key=lambda x: (x['bb_per_100'], x['wins']), reverse=True)
        return {'entrants': entrants, 'fixtures': fixtures, 'leaderboard': rows}
