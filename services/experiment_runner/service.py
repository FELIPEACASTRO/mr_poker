
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev

from packages.baseline_agent import BaselineAgent
from packages.engine import GameEngine
from packages.harness import MatchHarness


@dataclass
class ExperimentConfig:
    num_matches: int = 5
    hands_per_match: int = 100
    stacks: tuple[int, int] = (100, 100)
    seed_base: int = 20_000


class ExperimentRunner:
    def __init__(self, engine: GameEngine | None = None, report_dir: str = 'var/reports') -> None:
        self.engine = engine or GameEngine()
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def run_h2h(self, config: ExperimentConfig) -> dict:
        agent_a = BaselineAgent()
        agent_b = BaselineAgent()
        harness = MatchHarness(engine=self.engine, seat0_agent=agent_a, seat1_agent=agent_b)
        matches: list[dict] = []
        for idx in range(config.num_matches):
            result = harness.run(
                num_hands=config.hands_per_match,
                stacks=config.stacks,
                seed_base=config.seed_base + idx * 100_000,
            )
            result['match_index'] = idx
            matches.append(result)
        seat0_bb100 = [m['bb_per_100_seat0'] for m in matches]
        seat1_bb100 = [m['bb_per_100_seat1'] for m in matches]
        payload = {
            'experiment_type': 'h2h_eval',
            'config': asdict(config),
            'matches': matches,
            'summary': {
                'avg_bb_per_100_seat0': mean(seat0_bb100) if seat0_bb100 else 0.0,
                'avg_bb_per_100_seat1': mean(seat1_bb100) if seat1_bb100 else 0.0,
                'std_bb_per_100_seat0': pstdev(seat0_bb100) if len(seat0_bb100) > 1 else 0.0,
                'std_bb_per_100_seat1': pstdev(seat1_bb100) if len(seat1_bb100) > 1 else 0.0,
                'total_hands': config.num_matches * config.hands_per_match,
            },
        }
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        out = self.report_dir / f'experiment_{stamp}.json'
        out.write_text(json.dumps(payload, indent=2))
        payload['report_path'] = str(out)
        return payload
