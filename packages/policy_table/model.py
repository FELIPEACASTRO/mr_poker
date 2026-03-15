from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PolicyTableModel:
    model_id: str
    metadata: dict[str, Any]
    table: dict[str, dict[str, Any]]

    def predict(self, bucket_key: str, fallback: str = 'check') -> dict[str, Any]:
        row = self.table.get(bucket_key)
        if row is None:
            return {'action': fallback, 'confidence': 0.0, 'support': 0}
        return {'action': row['action'], 'confidence': row['confidence'], 'support': row['support']}
