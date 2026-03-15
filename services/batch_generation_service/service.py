from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from services.session_service import SessionConfig, SessionRunner


class BatchGenerationService:
    def __init__(self, session_runner: SessionRunner, output_dir: str = 'var/batches') -> None:
        self.session_runner = session_runner
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, *, batch_name: str = 'baseline_batch', num_hands: int = 50, seed_base: int = 90_000, stacks: tuple[int, int] = (100, 100)) -> dict:
        result = self.session_runner.run_h2h(SessionConfig(num_hands=num_hands, seed_base=seed_base, stacks=stacks, session_name=batch_name))
        batch_id = str(uuid4())
        payload = {
            'batch_id': batch_id,
            'batch_name': batch_name,
            'session_id': result['session_id'],
            'hands_played': result['hands_played'],
            'hand_ids': result['hand_ids'],
            'summary': result,
        }
        path = self.output_dir / f'{batch_id}.json'
        path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        payload['path'] = str(path)
        return payload
