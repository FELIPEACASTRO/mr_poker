from packages.common.types import ActionType, Street
from packages.engine import GameEngine


def test_start_new_hand_posts_blinds_and_deals_cards() -> None:
    engine = GameEngine()
    runtime = engine.start_new_hand(stacks=(100, 100), button_seat=0, seed=42)
    state = runtime.state

    assert state.street == Street.PRE_FLOP
    assert state.pot == 3
    assert state.acting_seat == 0
    assert state.to_call == 1
    assert len(state.players[0].hole_cards) == 2
    assert len(state.players[1].hole_cards) == 2
    assert state.players[0].stack == 99
    assert state.players[1].stack == 98
    assert state.actions[0].action_type == ActionType.POST_SMALL_BLIND
    assert state.actions[1].action_type == ActionType.POST_BIG_BLIND
