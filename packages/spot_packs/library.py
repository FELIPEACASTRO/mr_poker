
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ScriptedAction:
    action_type: str
    amount: int = 0


@dataclass(frozen=True)
class SpotPackScenario:
    spot_id: str
    name: str
    description: str
    objective: str
    stacks: tuple[int, int]
    button_seat: int
    deck_prefix: list[str]
    scripted_actions: list[ScriptedAction] = field(default_factory=list)
    expected_acting_seat: int | None = None
    tags: list[str] = field(default_factory=list)


SPOT_PACKS: dict[str, SpotPackScenario] = {
    'hu_preflop_open_value': SpotPackScenario(
        spot_id='hu_preflop_open_value',
        name='HU Preflop Open Value',
        description='Button on seat 0 with a premium pair facing the first decision of the hand.',
        objective='Validate opening logic, sizing and trace generation on a value-heavy preflop node.',
        stacks=(100, 100),
        button_seat=0,
        deck_prefix=['Ah', 'Kd', 'Ad', 'Qs'],
        scripted_actions=[],
        expected_acting_seat=0,
        tags=['preflop', 'open', 'value'],
    ),
    'hu_flop_cbet_ip': SpotPackScenario(
        spot_id='hu_flop_cbet_ip',
        name='HU Flop C-Bet In Position',
        description='Button completes preflop, big blind checks, flop checks to button on an ace-high board.',
        objective='Reach a simple flop continuation-bet node with clear initiative and board texture.',
        stacks=(100, 100),
        button_seat=0,
        deck_prefix=['As', 'Qh', 'Ks', 'Jh', '2c', 'Ac', '7s', '2d'],
        scripted_actions=[
            ScriptedAction('call', 1),
            ScriptedAction('check', 0),
            ScriptedAction('check', 0),
        ],
        expected_acting_seat=0,
        tags=['flop', 'cbet', 'in_position'],
    ),
    'hu_turn_barrel_value': SpotPackScenario(
        spot_id='hu_turn_barrel_value',
        name='HU Turn Barrel Value',
        description='Single-raised pot where button bet flop and got called, turn checks to button.',
        objective='Test turn barreling and value/protection logic on a dynamic board.',
        stacks=(100, 100),
        button_seat=0,
        deck_prefix=['Ac', 'Qh', 'Kc', 'Jh', '2s', 'Kd', '7h', '2d', '3c', 'Td'],
        scripted_actions=[
            ScriptedAction('call', 1),
            ScriptedAction('check', 0),
            ScriptedAction('check', 0),
            ScriptedAction('bet', 2),
            ScriptedAction('call', 2),
            ScriptedAction('check', 0),
        ],
        expected_acting_seat=0,
        tags=['turn', 'barrel', 'value'],
    ),
    'hu_river_bluffcatch': SpotPackScenario(
        spot_id='hu_river_bluffcatch',
        name='HU River Bluff Catcher',
        description='Single-raised pot reaches river after flop and turn checks; river decision is a bluff-catch node.',
        objective='Create a priced river node for fold/call calibration and session EV proxy reporting.',
        stacks=(100, 100),
        button_seat=0,
        deck_prefix=['Qd', 'Ah', 'Qs', 'Jh', '2s', 'Qc', '7h', '2d', '3c', '9d', '5s', 'Kh'],
        scripted_actions=[
            ScriptedAction('call', 1),
            ScriptedAction('check', 0),
            ScriptedAction('check', 0),
            ScriptedAction('check', 0),
            ScriptedAction('check', 0),
            ScriptedAction('check', 0),
        ],
        expected_acting_seat=1,
        tags=['river', 'bluffcatch', 'priced_node'],
    ),
}


def list_spot_packs() -> list[dict]:
    return [
        {
            "spot_id": s.spot_id,
            "name": s.name,
            "description": s.description,
            "objective": s.objective,
            "stacks": list(s.stacks),
            "button_seat": s.button_seat,
            "deck_prefix": list(s.deck_prefix),
            "scripted_actions": [{"action_type": a.action_type, "amount": a.amount} for a in s.scripted_actions],
            "expected_acting_seat": s.expected_acting_seat,
            "tags": list(s.tags),
        }
        for s in SPOT_PACKS.values()
    ]
