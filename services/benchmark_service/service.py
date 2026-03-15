
from __future__ import annotations

from packages.baseline_agent import BaselineAgent
from packages.engine import GameEngine
from packages.harness import MatchHarness
from services.experiment_runner.service import ExperimentConfig, ExperimentRunner


class BenchmarkService:
    def __init__(self, engine: GameEngine | None = None) -> None:
        self.engine = engine or GameEngine()

    def smoke(self, *, num_hands: int = 50, stacks: tuple[int, int] = (100, 100), seed_base: int = 10_000) -> dict:
        harness = MatchHarness(engine=self.engine)
        result = harness.run(num_hands=num_hands, stacks=stacks, seed_base=seed_base)
        result['benchmark_type'] = 'smoke_agent_vs_agent'
        return result

    def h2h_eval(self, *, num_matches: int = 3, hands_per_match: int = 50, stacks: tuple[int, int] = (100, 100), seed_base: int = 20_000) -> dict:
        runner = ExperimentRunner(engine=self.engine)
        payload = runner.run_h2h(ExperimentConfig(
            num_matches=num_matches,
            hands_per_match=hands_per_match,
            stacks=stacks,
            seed_base=seed_base,
        ))
        payload['benchmark_type'] = 'h2h_eval'
        return payload
