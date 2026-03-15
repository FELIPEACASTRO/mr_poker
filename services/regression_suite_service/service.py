from __future__ import annotations

import json
from pathlib import Path

from packages.spot_packs.library import list_spot_packs


class RegressionSuiteService:
    def __init__(self, golden_path: str = 'var/golden/golden_hands.json', output_dir: str = 'var/regression') -> None:
        self.golden_path = Path(golden_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

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
