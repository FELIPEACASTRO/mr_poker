from packages.common.types import ActionType
from packages.engine import GameEngine


def test_uncalled_all_in_is_returned_and_segmented() -> None:
    engine = GameEngine()
    runtime = engine.start_new_hand(stacks=(20, 10), button_seat=0, deck_prefix=["As", "Kd", "Ah", "Kc"])
    engine.apply_action(runtime, ActionType.ALL_IN)
    engine.apply_action(runtime, ActionType.CALL)

    snapshot = engine.state_snapshot(runtime)
    assert snapshot["is_terminal"] is True
    assert snapshot["players"][0]["total_invested"] == 10
    assert snapshot["players"][1]["total_invested"] == 10
    assert snapshot["pot_segments"][0]["amount"] == 20
    assert any(action["note"] == "uncalled_return" for action in snapshot["actions"])
    assert snapshot["players"][0]["stack"] + snapshot["players"][1]["stack"] == 30
