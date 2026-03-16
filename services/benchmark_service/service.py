
from __future__ import annotations

import time
from statistics import mean, pstdev
from typing import Any

from packages.engine import GameEngine
from packages.harness import MatchHarness
from services.experiment_runner.service import ExperimentConfig, ExperimentRunner


class BenchmarkService:
    def __init__(self, engine: GameEngine | None = None, warmup_hands: int = 5) -> None:
        self.engine = engine or GameEngine()
        self.warmup_hands = warmup_hands

    def smoke(self, *, num_hands: int = 50, stacks: tuple[int, int] = (100, 100), seed_base: int = 10_000) -> dict:
        harness = MatchHarness(engine=self.engine)
        if self.warmup_hands > 0:
            harness.run(num_hands=self.warmup_hands, stacks=stacks, seed_base=seed_base - 1000)
        start = time.monotonic()
        result = harness.run(num_hands=num_hands, stacks=stacks, seed_base=seed_base)
        elapsed = time.monotonic() - start
        result['benchmark_type'] = 'smoke_agent_vs_agent'
        result['elapsed_seconds'] = round(elapsed, 3)
        result['hands_per_second'] = round(num_hands / elapsed, 2) if elapsed > 0 else 0
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

    def regression_check(
        self,
        *,
        baseline_bb100: float = 0.0,
        num_matches: int = 5,
        hands_per_match: int = 100,
        seed_base: int = 30_000,
        stacks: tuple[int, int] = (100, 100),
    ) -> dict[str, Any]:
        runner = ExperimentRunner(engine=self.engine)
        payload = runner.run_h2h(ExperimentConfig(
            num_matches=num_matches,
            hands_per_match=hands_per_match,
            stacks=stacks,
            seed_base=seed_base,
        ))
        matches = payload.get('matches', [])
        seat0_bb100 = [m['bb_per_100_seat0'] for m in matches]
        current_avg = mean(seat0_bb100) if seat0_bb100 else 0.0
        std = pstdev(seat0_bb100) if len(seat0_bb100) > 1 else 0.0
        regression_detected = current_avg < baseline_bb100 - 2 * std if std > 0 else current_avg < baseline_bb100
        return {
            'benchmark_type': 'regression_check',
            'baseline_bb100': baseline_bb100,
            'current_avg_bb100': round(current_avg, 4),
            'std_bb100': round(std, 4),
            'regression_detected': regression_detected,
            'num_matches': num_matches,
            'hands_per_match': hands_per_match,
            'detail': payload,
        }
