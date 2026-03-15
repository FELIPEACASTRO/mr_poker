from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ExternalSolverAdapter:
    def __init__(self, base_dir: str = 'var/external_solver') -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def catalog(self) -> list[dict[str, Any]]:
        result = []
        for path in sorted(self.base_dir.glob('*.json')):
            payload = json.loads(path.read_text(encoding='utf-8'))
            result.append({'name': payload.get('name', path.stem), 'path': str(path), 'label_count': len(payload.get('labels', []))})
        return result

    def load(self, name: str) -> dict[str, Any]:
        path = self.base_dir / f'{name}.json'
        if not path.exists():
            raise KeyError(name)
        return json.loads(path.read_text(encoding='utf-8'))

    def compare(self, name: str, *, spot_id: str | None = None, bucket_key: str | None = None) -> dict[str, Any]:
        payload = self.load(name)
        for label in payload.get('labels', []):
            if spot_id is not None and label.get('spot_id') == spot_id:
                return {'matched_on': 'spot_id', 'label': label, 'solver_name': payload.get('name', name)}
            if bucket_key is not None and label.get('bucket_key') == bucket_key:
                return {'matched_on': 'bucket_key', 'label': label, 'solver_name': payload.get('name', name)}
        return {'matched_on': None, 'label': None, 'solver_name': payload.get('name', name)}
