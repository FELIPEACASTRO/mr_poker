from __future__ import annotations

from typing import Any

from packages.policy_table import PolicyRegistry, PolicyTableTrainer
from packages.solver_like import bucketize_spot
from services.solver_label_service import SolverLabelService


class ModelService:
    def __init__(self, solver_labels: SolverLabelService, model_dir: str = 'var/models') -> None:
        self.solver_labels = solver_labels
        self.registry = PolicyRegistry(model_dir)
        self.trainer = PolicyTableTrainer()

    def train_policy_table(self) -> dict[str, Any]:
        rows = self.solver_labels.build_training_rows_from_spot_packs()
        model = self.trainer.train(rows, model_name='policy_table_sprint08')
        saved = self.registry.save(model)
        saved['training_rows'] = len(rows)
        saved['bucket_count'] = model.metadata['bucket_count']
        return saved

    def list_models(self) -> list[dict[str, Any]]:
        return self.registry.list_models()

    def evaluate_model_on_spot_packs(self, model_id: str) -> dict[str, Any]:
        model = self.registry.load(model_id)
        rows = self.solver_labels.build_training_rows_from_spot_packs()
        results = []
        matches = 0
        for row in rows:
            bucket_key = (row.get('bucket_info') or bucketize_spot(row))['bucket_key']
            pred = model.predict(bucket_key, fallback='check')
            aligned = pred['action'] == row['label_action']
            matches += int(aligned)
            results.append({
                'source': row['source'],
                'predicted_action': pred['action'],
                'label_action': row['label_action'],
                'aligned': aligned,
                'support': pred['support'],
            })
        return {
            'model_id': model_id,
            'rows': len(rows),
            'alignment_rate': round(matches / len(rows), 4) if rows else 0.0,
            'results': results,
        }
