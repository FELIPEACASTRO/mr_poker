
from packages.baseline_agent import BaselineAgent
from packages.engine import GameEngine


def test_baseline_agent_returns_legal_action() -> None:
    engine = GameEngine()
    runtime = engine.start_new_hand(stacks=(100, 100), button_seat=0, deck_prefix=['As', 'Ah', '2c', '7d'])
    decision = BaselineAgent().decide(runtime, engine)
    assert decision.action_type in engine.legal_actions(runtime)
