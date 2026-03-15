from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean
from typing import Any
from uuid import uuid4

from packages.solver_like import bucketize_spot
from .model import PolicyTableModel


class PolicyTableTrainer:
    def train(self, labeled_spots: list[dict[str, Any]], *, model_name: str = 'policy_table_v1') -> PolicyTableModel:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for spot in labeled_spots:
            bucket_info = spot.get('bucket_info') or bucketize_spot(spot)
            grouped[bucket_info['bucket_key']].append({**spot, 'bucket_info': bucket_info})
        table: dict[str, dict[str, Any]] = {}
        for bucket_key, rows in grouped.items():
            counts = Counter(str(r['label_action']) for r in rows)
            action, support = counts.most_common(1)[0]
            table[bucket_key] = {
                'action': action,
                'support': support,
                'confidence': round(support / len(rows), 4),
                'mean_equity': round(mean(float(r.get('estimated_equity', 0.0)) for r in rows), 4),
                'mean_pot_odds': round(mean(float(r.get('pot_odds', 0.0)) for r in rows), 4),
            }
        metadata = {
            'model_name': model_name,
            'row_count': len(labeled_spots),
            'bucket_count': len(table),
            'trainer': 'PolicyTableTrainer',
        }
        return PolicyTableModel(model_id=str(uuid4()), metadata=metadata, table=table)
