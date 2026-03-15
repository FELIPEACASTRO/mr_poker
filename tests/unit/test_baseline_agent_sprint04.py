
from packages.baseline_agent import BaselineAgent
from packages.common.types import ActionType
from packages.engine import GameEngine


def test_baseline_agent_is_aggressive_with_premium_preflop() -> None:
    engine = GameEngine()
    runtime = engine.start_new_hand(stacks=(100, 100), button_seat=0, deck_prefix=['As', '2c', 'Ah', '7d'])
    decision = BaselineAgent().decide(runtime, engine)
    assert decision.action_type in {ActionType.BET, ActionType.RAISE, ActionType.ALL_IN}
    assert decision.trace is not None
    assert decision.trace.estimated_equity > 0.6
