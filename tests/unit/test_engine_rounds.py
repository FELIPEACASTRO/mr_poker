from packages.common.types import ActionType, Street
from packages.engine import GameEngine


def test_preflop_call_then_big_blind_check_advances_to_flop() -> None:
    engine = GameEngine()
    runtime = engine.start_new_hand(stacks=(100, 100), button_seat=0, seed=7)

    engine.apply_action(runtime, ActionType.CALL)
    assert runtime.state.street == Street.PRE_FLOP
    assert runtime.state.acting_seat == 1
    assert runtime.state.to_call == 0

    engine.apply_action(runtime, ActionType.CHECK)
    state = runtime.state
    assert state.street == Street.FLOP
    assert len(state.board) == 3
    assert state.acting_seat == 1
    assert state.to_call == 0


def test_flop_double_check_advances_turn() -> None:
    engine = GameEngine()
    runtime = engine.start_new_hand(stacks=(100, 100), button_seat=0, seed=7)
    engine.apply_action(runtime, ActionType.CALL)
    engine.apply_action(runtime, ActionType.CHECK)

    engine.apply_action(runtime, ActionType.CHECK)
    assert runtime.state.street == Street.FLOP
    assert runtime.state.acting_seat == 0

    engine.apply_action(runtime, ActionType.CHECK)
    assert runtime.state.street == Street.TURN
    assert len(runtime.state.board) == 4
    assert runtime.state.acting_seat == 1


def test_preflop_fold_awards_pot_to_big_blind() -> None:
    engine = GameEngine()
    runtime = engine.start_new_hand(stacks=(100, 100), button_seat=0, seed=7)
    engine.apply_action(runtime, ActionType.FOLD)
    state = runtime.state

    assert state.is_terminal is True
    assert state.winner_seat == 1
    assert state.players[0].stack == 99
    assert state.players[1].stack == 101
