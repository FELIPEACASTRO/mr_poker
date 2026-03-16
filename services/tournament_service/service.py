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
        if name.startswith('llm:'):
            from packages.llm_agent import LLMPokerAgent
            return LLMPokerAgent(model_id=name.split(':', 1)[1])
        if name.startswith('cfr:'):
            from packages.cfr_agent import CFRAgent
            from packages.cfr_agent.trainer import CFRState
            cfr_path = name.split(':', 1)[1]
            return CFRAgent(CFRState.load(cfr_path))
        if name == 'cfr':
            from packages.cfr_agent import CFRAgent, CFRTrainer
            trainer = CFRTrainer(
                small_blind=self.engine.small_blind,
                big_blind=self.engine.big_blind,
            )
            state = trainer.train(iterations=5000)
            return CFRAgent(state)
        raise KeyError(name)

    def round_robin(self, entrants: list[str], *, hands_per_match: int = 30, seed_base: int = 70_000, stacks: tuple[int, int] = (100, 100)) -> dict[str, Any]:
        leaderboard: dict[str, dict[str, Any]] = defaultdict(lambda: {'matches': 0, 'hands': 0, 'profit': 0, 'wins': 0, 'opponents_beaten': 0})
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
                # SOS tiebreaker: track opponents beaten
                if result['seat0_profit'] > result['seat1_profit']:
                    leaderboard[seat0_name]['opponents_beaten'] += 1
                elif result['seat1_profit'] > result['seat0_profit']:
                    leaderboard[seat1_name]['opponents_beaten'] += 1
        rows = []
        for entrant, stats in leaderboard.items():
            hands = stats['hands']
            bb100 = round((stats['profit'] / self.engine.big_blind) * (100 / hands), 4) if hands else 0.0
            rows.append({
                'entrant': entrant,
                'matches': stats['matches'],
                'hands': hands,
                'profit': stats['profit'],
                'wins': stats['wins'],
                'bb_per_100': bb100,
                'opponents_beaten': stats['opponents_beaten'],
            })
        # Sort by bb/100, then by opponents_beaten (Buchholz-like tiebreak), then wins
        rows.sort(key=lambda x: (x['bb_per_100'], x['opponents_beaten'], x['wins']), reverse=True)
        return {'entrants': entrants, 'fixtures': fixtures, 'leaderboard': rows}

    def knockout(self, entrants: list[str], *, hands_per_match: int = 50, seed_base: int = 80_000, stacks: tuple[int, int] = (100, 100)) -> dict[str, Any]:
        """Single-elimination knockout tournament."""
        if len(entrants) < 2:
            return {'error': 'Need at least 2 entrants', 'entrants': entrants}
        rounds: list[list[dict[str, Any]]] = []
        remaining = list(entrants)
        seed = seed_base
        round_num = 0
        while len(remaining) > 1:
            round_num += 1
            round_fixtures = []
            next_round = []
            # If odd number, last gets a bye
            pairs = []
            for i in range(0, len(remaining) - 1, 2):
                pairs.append((remaining[i], remaining[i + 1]))
            bye = remaining[-1] if len(remaining) % 2 == 1 else None

            for seat0_name, seat1_name in pairs:
                harness = MatchHarness(
                    engine=self.engine,
                    seat0_agent=self._agent_factory(seat0_name),
                    seat1_agent=self._agent_factory(seat1_name),
                )
                result = harness.run(num_hands=hands_per_match, stacks=stacks, seed_base=seed)
                seed += hands_per_match + 17
                winner = seat0_name if result['seat0_profit'] >= result['seat1_profit'] else seat1_name
                round_fixtures.append({
                    'seat0': seat0_name,
                    'seat1': seat1_name,
                    'winner': winner,
                    'result': result,
                })
                next_round.append(winner)
            if bye:
                round_fixtures.append({'seat0': bye, 'seat1': None, 'winner': bye, 'result': 'bye'})
                next_round.append(bye)
            rounds.append(round_fixtures)
            remaining = next_round

        return {
            'entrants': entrants,
            'champion': remaining[0] if remaining else None,
            'rounds': rounds,
        }
