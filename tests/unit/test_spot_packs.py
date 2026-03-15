
from packages.engine import GameEngine
from packages.persistence import SqliteHandStore
from services.spot_pack_service import SpotPackService


def test_spot_pack_instantiation_reaches_expected_actor(tmp_path) -> None:
    service = SpotPackService(engine=GameEngine(), store=SqliteHandStore(str(tmp_path / 'spot.db')))
    runtime, pack = service.instantiate('hu_flop_cbet_ip')
    assert runtime.state.acting_seat == pack['expected_acting_seat'] == 0
    assert runtime.state.street.value == 'flop'
    assert len(runtime.state.board) == 3
