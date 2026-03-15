from __future__ import annotations

from typing import Any

from packages.adaptive_agent import AdaptiveBaselineAgent
from packages.baseline_agent import BaselineAgent
from packages.harness import MatchHarness
from packages.opponent_model import build_opponent_profile
from packages.persistence import SqliteHandStore
from packages.engine import GameEngine


class OpponentProfileService:
    def __init__(self, store: SqliteHandStore, engine: GameEngine | None = None) -> None:
        self.store = store
        self.engine = engine or GameEngine()

    def session_profiles(self, session_id: str) -> dict[str, Any]:
        hands = self.store.get_hands_for_session(session_id)
        if not hands:
            raise KeyError(session_id)
        aggregate = []
        for hand in hands:
            aggregate.extend(self.store.get_actions(hand['hand_id']))
        return {
            'session_id': session_id,
            'profiles': {
                'seat0': build_opponent_profile(aggregate, seat=0),
                'seat1': build_opponent_profile(aggregate, seat=1),
            },
        }

    def adaptive_benchmark(self, *, num_hands: int = 40, stacks: tuple[int, int] = (100, 100), seed_base: int = 50_000) -> dict[str, Any]:
        harness = MatchHarness(engine=self.engine, seat0_agent=AdaptiveBaselineAgent(), seat1_agent=BaselineAgent())
        result = harness.run(num_hands=num_hands, stacks=stacks, seed_base=seed_base)
        result['benchmark_type'] = 'adaptive_vs_baseline'
        return result
