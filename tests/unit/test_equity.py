
from packages.engine.models import Card
from packages.equity import estimate_equity


def test_equity_estimator_returns_reasonable_range_for_aces_preflop() -> None:
    equity = estimate_equity([Card.from_str('As'), Card.from_str('Ah')], [], samples=300, seed=123)
    assert 0.72 <= equity <= 0.95
