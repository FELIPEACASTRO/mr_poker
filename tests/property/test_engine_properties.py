"""Property-based tests for the poker engine.

Tests fundamental invariants across many random seeds using standard pytest.
"""
from __future__ import annotations

import random

import pytest

from packages.common.types import ActionType
from packages.engine.engine import GameEngine, HandRuntime


def _play_random_hand(engine: GameEngine, seed: int, stacks: tuple[int, ...] = (100, 100)) -> HandRuntime:
    """Play a hand to completion with random legal actions."""
    rng = random.Random(seed)
    runtime = engine.start_new_hand(stacks=stacks, button_seat=0, seed=seed)
    max_actions = 200  # safety limit
    for _ in range(max_actions):
        if runtime.state.is_terminal:
            break
        actions = engine.legal_actions(runtime)
        if not actions:
            break
        action = rng.choice(actions)
        amount = 0
        if action in (ActionType.BET, ActionType.RAISE):
            amount = (runtime.state.min_raise_to or runtime.state.current_bet + engine.big_blind)
        engine.apply_action(runtime, action, amount)
    return runtime


# ---------------------------------------------------------------------------
# Property 1: Chip conservation
# ---------------------------------------------------------------------------
SEEDS = list(range(50))


@pytest.mark.parametrize("seed", SEEDS)
def test_chip_conservation(seed: int) -> None:
    """Sum of all player stacks at end == sum at start."""
    engine = GameEngine()
    initial_total = 200  # stacks=(100, 100)
    runtime = _play_random_hand(engine, seed)
    state = runtime.state
    final_total = sum(p.stack for p in state.players.values()) + state.pot
    assert final_total == initial_total, (
        f"seed={seed}: initial={initial_total} final_stacks+pot={final_total}"
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_chip_conservation_3_players(seed: int) -> None:
    """Chip conservation holds with 3 players."""
    engine = GameEngine()
    stacks = (100, 100, 100)
    initial_total = sum(stacks)
    runtime = _play_random_hand(engine, seed, stacks=stacks)
    state = runtime.state
    final_total = sum(p.stack for p in state.players.values()) + state.pot
    assert final_total == initial_total, (
        f"seed={seed}: initial={initial_total} final_stacks+pot={final_total}"
    )


# ---------------------------------------------------------------------------
# Property 2: Terminal state has winner
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", SEEDS)
def test_terminal_has_winner(seed: int) -> None:
    """If is_terminal, winner_seat is not None or showdown was reached (split pot)."""
    engine = GameEngine()
    runtime = _play_random_hand(engine, seed)
    state = runtime.state
    if state.is_terminal:
        # Either there is a winner, or it was a showdown (possible split)
        assert state.winner_seat is not None or state.showdown_reached, (
            f"seed={seed}: terminal but no winner and no showdown"
        )


# ---------------------------------------------------------------------------
# Property 3: Pot is always non-negative
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", SEEDS)
def test_pot_non_negative(seed: int) -> None:
    """Pot should never go negative during or after play."""
    engine = GameEngine()
    rng = random.Random(seed)
    runtime = engine.start_new_hand(stacks=(100, 100), button_seat=0, seed=seed)
    assert runtime.state.pot >= 0, f"seed={seed}: pot negative after start"
    for _ in range(200):
        if runtime.state.is_terminal:
            break
        actions = engine.legal_actions(runtime)
        if not actions:
            break
        action = rng.choice(actions)
        amount = 0
        if action in (ActionType.BET, ActionType.RAISE):
            amount = (runtime.state.min_raise_to or runtime.state.current_bet + engine.big_blind)
        engine.apply_action(runtime, action, amount)
        assert runtime.state.pot >= 0, (
            f"seed={seed}: pot={runtime.state.pot} went negative"
        )


# ---------------------------------------------------------------------------
# Property 4: No player has negative stack at any point
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", SEEDS)
def test_no_negative_stacks(seed: int) -> None:
    """No player should ever have a negative stack."""
    engine = GameEngine()
    rng = random.Random(seed)
    runtime = engine.start_new_hand(stacks=(100, 100), button_seat=0, seed=seed)
    for p in runtime.state.players.values():
        assert p.stack >= 0, f"seed={seed}: player {p.seat} negative stack after start"
    for _ in range(200):
        if runtime.state.is_terminal:
            break
        actions = engine.legal_actions(runtime)
        if not actions:
            break
        action = rng.choice(actions)
        amount = 0
        if action in (ActionType.BET, ActionType.RAISE):
            amount = (runtime.state.min_raise_to or runtime.state.current_bet + engine.big_blind)
        engine.apply_action(runtime, action, amount)
        for p in runtime.state.players.values():
            assert p.stack >= 0, (
                f"seed={seed}: player {p.seat} has stack={p.stack}"
            )


# ---------------------------------------------------------------------------
# Property 5: Legal actions never empty when not terminal and acting_seat set
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", SEEDS)
def test_legal_actions_not_empty_when_active(seed: int) -> None:
    """When hand is not terminal and acting_seat is set, legal_actions must be non-empty."""
    engine = GameEngine()
    rng = random.Random(seed)
    runtime = engine.start_new_hand(stacks=(100, 100), button_seat=0, seed=seed)
    for _ in range(200):
        if runtime.state.is_terminal:
            break
        if runtime.state.acting_seat is not None:
            actions = engine.legal_actions(runtime)
            assert len(actions) > 0, (
                f"seed={seed}: acting_seat={runtime.state.acting_seat} but no legal actions"
            )
            action = rng.choice(actions)
            amount = 0
            if action in (ActionType.BET, ActionType.RAISE):
                amount = (runtime.state.min_raise_to or runtime.state.current_bet + engine.big_blind)
            engine.apply_action(runtime, action, amount)
        else:
            break


# ---------------------------------------------------------------------------
# Property 6: validate_invariants returns empty for completed hands
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", SEEDS)
def test_validate_invariants_clean(seed: int) -> None:
    """validate_invariants should report no violations for properly played hands."""
    engine = GameEngine()
    runtime = _play_random_hand(engine, seed)
    violations = engine.validate_invariants(runtime)
    assert violations == [], f"seed={seed}: violations={violations}"


# ---------------------------------------------------------------------------
# Multi-player property tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", list(range(30)))
@pytest.mark.parametrize("num_players", [2, 3, 4, 5, 6])
def test_chip_conservation_n_players(seed: int, num_players: int) -> None:
    """Chip conservation with N players."""
    engine = GameEngine()
    stacks = tuple([100] * num_players)
    initial_total = sum(stacks)
    runtime = _play_random_hand(engine, seed, stacks=stacks)
    state = runtime.state
    final_total = sum(p.stack for p in state.players.values()) + state.pot
    assert final_total == initial_total, (
        f"seed={seed}, n={num_players}: initial={initial_total} final={final_total}"
    )
