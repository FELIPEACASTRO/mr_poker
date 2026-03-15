from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .model import PolicyTableModel


class PolicyRegistry:
    def __init__(self, base_dir: str = 'var/models') -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save(self, model: PolicyTableModel) -> dict[str, Any]:
        path = self.base_dir / f'{model.model_id}.json'
        payload = {'model_id': model.model_id, 'metadata': model.metadata, 'table': model.table}
        path.write_text(json.dumps(payload, indent=2))
        return {'model_id': model.model_id, 'path': str(path), 'metadata': model.metadata}

    def load(self, model_id: str) -> PolicyTableModel:
        path = self.base_dir / f'{model_id}.json'
        payload = json.loads(path.read_text())
        return PolicyTableModel(model_id=payload['model_id'], metadata=payload['metadata'], table=payload['table'])

    def list_models(self) -> list[dict[str, Any]]:
        result = []
        for path in sorted(self.base_dir.glob('*.json')):
            payload = json.loads(path.read_text())
            result.append({'model_id': payload['model_id'], 'metadata': payload['metadata'], 'path': str(path)})
        return result
