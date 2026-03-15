import json
from pathlib import Path

from packages.common.types import ActionType
from packages.engine import GameEngine


def test_golden_hands_regression() -> None:
    fixture_path = Path("var/golden/golden_hands.json")
    cases = json.loads(fixture_path.read_text())
    engine = GameEngine()
    for case in cases:
        runtime = engine.start_new_hand(
            stacks=tuple(case["stacks"]),
            button_seat=case["button_seat"],
            deck_prefix=case["deck_prefix"],
        )
        for step in case["actions"]:
            engine.apply_action(runtime, ActionType(step["action_type"]), int(step["amount"]))
        snapshot = engine.state_snapshot(runtime)
        assert snapshot["is_terminal"] == case["expected"]["terminal"], case["name"]
        assert snapshot["winner_seat"] == case["expected"]["winner_seat"], case["name"]
        assert snapshot["players"][0]["stack"] == case["expected"]["seat0_stack"], case["name"]
        assert snapshot["players"][1]["stack"] == case["expected"]["seat1_stack"], case["name"]
