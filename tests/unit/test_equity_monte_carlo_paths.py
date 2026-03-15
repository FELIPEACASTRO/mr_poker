from __future__ import annotations

import pytest

from packages.engine.models import Card
from packages.equity import monte_carlo as mc


class _FakeEval7:
    @staticmethod
    def Card(card_repr: str) -> str:
        return card_repr

    @staticmethod
    def evaluate(cards: list[str]) -> int:
        # Deterministic and fast pseudo-score for test coverage only.
        return sum((ord(card[0]) * 3) + ord(card[1]) for card in cards)


def _clear_mc_caches() -> None:
    mc._eval7_card.cache_clear()
    mc._estimate_equity_cached.cache_clear()


def test_eval7_helpers_raise_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mc, "eval7", None)
    _clear_mc_caches()

    with pytest.raises(RuntimeError, match="eval7 unavailable"):
        mc._eval7_card("As")

    with pytest.raises(RuntimeError, match="eval7 unavailable"):
        mc._score_eval7(("As", "Ah"))


def test_cached_estimator_validates_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mc, "eval7", None)
    _clear_mc_caches()

    with pytest.raises(ValueError, match="exactly 1 opponent"):
        mc._estimate_equity_cached(("As", "Ah"), tuple(), 2, 16, 7)

    with pytest.raises(ValueError, match="exactly 2 cards"):
        mc._estimate_equity_cached(("As",), tuple(), 1, 16, 7)

    with pytest.raises(ValueError, match="board cannot exceed 5 cards"):
        mc._estimate_equity_cached(
            ("As", "Ah"), ("2c", "3d", "4h", "5s", "6c", "7d"), 1, 16, 7
        )


def test_eval7_equity_paths_for_river_turn_and_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mc, "eval7", _FakeEval7)
    _clear_mc_caches()

    hero = ("As", "Ah")

    # River path (exact opponent combinations, len(board)=5).
    monkeypatch.setattr(
        mc,
        "FULL_DECK_STR",
        (
            "As",
            "Ah",
            "2c",
            "3c",
            "4c",
            "5c",
            "6c",
            "7d",
            "8d",
        ),
    )
    river_eq = mc._estimate_equity_eval7(
        hero_hole=hero,
        board=("2c", "3c", "4c", "5c", "6c"),
        samples=8,
        seed=11,
    )
    assert 0.0 <= river_eq <= 1.0

    # Turn path (exact river+opponent enumeration, len(board)=4).
    monkeypatch.setattr(
        mc,
        "FULL_DECK_STR",
        (
            "As",
            "Ah",
            "2c",
            "3c",
            "4c",
            "5c",
            "7d",
            "8d",
            "9d",
            "Td",
        ),
    )
    turn_eq = mc._estimate_equity_eval7(
        hero_hole=hero,
        board=("2c", "3c", "4c", "5c"),
        samples=8,
        seed=11,
    )
    assert 0.0 <= turn_eq <= 1.0

    # Sampling path (flop/preflop style, len(board)<4).
    monkeypatch.setattr(
        mc,
        "FULL_DECK_STR",
        (
            "As",
            "Ah",
            "2c",
            "3c",
            "4c",
            "5c",
            "6c",
            "7d",
            "8d",
            "9d",
            "Td",
            "Jd",
        ),
    )
    sampled_eq = mc._estimate_equity_eval7(
        hero_hole=hero,
        board=tuple(),
        samples=4,
        seed=11,
    )
    assert 0.0 <= sampled_eq <= 1.0

    # Cached entry point with eval7 enabled (covers eval7 dispatch branch).
    _clear_mc_caches()
    cached_eq = mc._estimate_equity_cached(hero, tuple(), 1, 4, 11)
    assert 0.0 <= cached_eq <= 1.0


def test_estimate_equity_is_order_invariant_and_uses_canonical_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mc, "eval7", None)
    _clear_mc_caches()

    eq_a = mc.estimate_equity(
        [Card.from_str("As"), Card.from_str("Ah")],
        [Card.from_str("2c"), Card.from_str("3d"), Card.from_str("4h")],
        samples=24,
        seed=99,
    )
    eq_b = mc.estimate_equity(
        [Card.from_str("Ah"), Card.from_str("As")],
        [Card.from_str("4h"), Card.from_str("2c"), Card.from_str("3d")],
        samples=24,
        seed=99,
    )
    assert eq_a == eq_b


def test_eval7_tie_branches_are_accounted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mc, "eval7", _FakeEval7)
    _clear_mc_caches()

    # Force tie behavior in all eval7 paths.
    monkeypatch.setattr(mc, "_score_eval7", lambda cards: 100)

    hero = ("As", "Ah")

    monkeypatch.setattr(
        mc,
        "FULL_DECK_STR",
        ("As", "Ah", "2c", "3c", "4c", "5c", "6c", "7d", "8d"),
    )
    eq_river = mc._estimate_equity_eval7(
        hero_hole=hero,
        board=("2c", "3c", "4c", "5c", "6c"),
        samples=4,
        seed=1,
    )
    assert eq_river == 0.5

    monkeypatch.setattr(
        mc,
        "FULL_DECK_STR",
        ("As", "Ah", "2c", "3c", "4c", "5c", "7d", "8d", "9d", "Td"),
    )
    eq_turn = mc._estimate_equity_eval7(
        hero_hole=hero,
        board=("2c", "3c", "4c", "5c"),
        samples=4,
        seed=1,
    )
    assert eq_turn == 0.5

    monkeypatch.setattr(
        mc,
        "FULL_DECK_STR",
        ("As", "Ah", "2c", "3c", "4c", "5c", "6c", "7d", "8d", "9d", "Td", "Jd"),
    )
    eq_sample = mc._estimate_equity_eval7(
        hero_hole=hero,
        board=tuple(),
        samples=3,
        seed=1,
    )
    assert eq_sample == 0.5


def test_python_board_and_sampling_paths_cover_ties(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mc, "eval7", None)
    _clear_mc_caches()

    # Force tie behavior in python evaluator path.
    monkeypatch.setattr(mc, "_score_python", lambda cards: (4, (10,)))

    hero = ("As", "Ah")

    monkeypatch.setattr(
        mc,
        "FULL_DECK_STR",
        ("As", "Ah", "2c", "3c", "4c", "5c", "6c", "7d", "8d"),
    )
    eq_river = mc._estimate_equity_python(
        hero_hole=hero,
        board=("2c", "3c", "4c", "5c", "6c"),
        samples=4,
        seed=1,
    )
    assert eq_river == 0.5

    monkeypatch.setattr(
        mc,
        "FULL_DECK_STR",
        ("As", "Ah", "2c", "3c", "4c", "5c", "6c", "7d", "8d", "9d", "Td", "Jd"),
    )
    eq_sample = mc._estimate_equity_python(
        hero_hole=hero,
        board=("2c", "3c", "4c"),
        samples=4,
        seed=1,
    )
    assert eq_sample == 0.5
