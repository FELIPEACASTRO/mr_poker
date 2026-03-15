from packages.harness import MatchHarness


def test_agent_vs_agent_harness_is_zero_sum() -> None:
    result = MatchHarness().run(num_hands=12, seed_base=200)
    assert result["hands_played"] == 12
    assert result["seat0_profit"] + result["seat1_profit"] == 0
    assert result["avg_actions_per_hand"] > 0
