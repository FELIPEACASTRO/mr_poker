from packages.common.types import Street
from packages.engine.models import HandState, PlayerState
from packages.features.minimum import derive_minimum_features


def test_minimum_features() -> None:
    player = PlayerState(seat=0, stack=100)
    state = HandState(
        hand_id="h1",
        button_seat=0,
        street=Street.PRE_FLOP,
        pot=3,
        to_call=1,
        min_raise_to=4,
        players={0: player},
    )
    features = derive_minimum_features(state, player)
    assert features.pot == 3
    assert features.to_call == 1
    assert features.board_size == 0
