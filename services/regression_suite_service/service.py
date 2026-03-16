from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from packages.common.types import ActionType
from packages.engine.engine import GameEngine
from packages.spot_packs.library import list_spot_packs


class RegressionSuiteService:
    def __init__(
        self,
        golden_path: str = 'var/golden/golden_hands.json',
        output_dir: str = 'var/regression',
        engine: GameEngine | None = None,
    ) -> None:
        self.golden_path = Path(golden_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.engine = engine or GameEngine()

    def build(self, suite_name: str = 'core_regression_v1') -> dict:
        golden_hands = []
        if self.golden_path.exists():
            golden_hands = json.loads(self.golden_path.read_text(encoding='utf-8'))
        spots = list_spot_packs()
        payload = {
            'suite_name': suite_name,
            'golden_hand_count': len(golden_hands),
            'golden_hand_ids': [item.get('hand_id') for item in golden_hands],
            'spot_pack_count': len(spots),
            'spot_pack_ids': [item['spot_id'] for item in spots],
            'recommended_gate': {
                'golden_hands_all_pass': True,
                'spot_pack_solver_like_match_rate_min': 0.55,
            },
        }
        path = self.output_dir / f'{suite_name}.json'
        path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        payload['path'] = str(path)
        return payload

    def run_golden_hands(self) -> dict[str, Any]:
        """Replay golden hands and verify they produce expected outcomes."""
        if not self.golden_path.exists():
            return {'status': 'skipped', 'reason': 'golden_hands.json not found'}
        golden_hands = json.loads(self.golden_path.read_text(encoding='utf-8'))
        results = []
        passed = 0
        for item in golden_hands:
            hand_id = item.get('hand_id', 'unknown')
            try:
                stacks = tuple(item.get('stacks', (100, 100)))
                runtime = self.engine.start_new_hand(
                    stacks=stacks,
                    button_seat=item.get('button_seat', 0),
                    seed=item.get('seed'),
                    deck_prefix=item.get('deck_prefix', []),
                    hand_id=hand_id,
                )
                for action in item.get('actions', []):
                    self.engine.apply_action(
                        runtime,
                        ActionType(action['action_type']),
                        int(action.get('amount', 0)),
                    )
                ok = runtime.state.is_terminal == item.get('expected_terminal', True)
                if ok and 'expected_winner' in item:
                    ok = runtime.state.winner_seat == item['expected_winner']
                violations = self.engine.validate_invariants(runtime)
                ok = ok and len(violations) == 0
                passed += int(ok)
                results.append({'hand_id': hand_id, 'passed': ok, 'violations': violations})
            except Exception as exc:
                results.append({'hand_id': hand_id, 'passed': False, 'error': str(exc)})
        return {
            'total': len(golden_hands),
            'passed': passed,
            'failed': len(golden_hands) - passed,
            'results': results,
        }
