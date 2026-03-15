from packages.common.types import ActionType
from packages.engine import GameEngine
from packages.evaluator.hands import best_hand_rank, hand_label
from packages.engine.models import Card


def test_pair_of_aces_beats_high_card_after_checkdown() -> None:
    engine = GameEngine()
    runtime = engine.start_new_hand(
        stacks=(100, 100),
        button_seat=0,
        deck_prefix=[
            "Ah", "Kc", "Ad", "Qc", "5c", "2s", "7d", "9h", "6c", "Jc", "8c", "3d"
        ],
    )

    engine.apply_action(runtime, ActionType.CALL)
    engine.apply_action(runtime, ActionType.CHECK)
    engine.apply_action(runtime, ActionType.CHECK)
    engine.apply_action(runtime, ActionType.CHECK)
    engine.apply_action(runtime, ActionType.CHECK)
    engine.apply_action(runtime, ActionType.CHECK)
    engine.apply_action(runtime, ActionType.CHECK)
    engine.apply_action(runtime, ActionType.CHECK)

    state = runtime.state
    assert state.is_terminal is True
    assert state.winner_seat == 0
    assert state.players[0].stack == 102
    assert state.players[1].stack == 98


def test_hand_label_for_full_house() -> None:
    cards = [Card.from_str(v) for v in ["Ah", "Ad", "Ac", "Kh", "Kd", "2s", "3c"]]
    rank = best_hand_rank(cards)
    assert hand_label(rank) == "full_house"
