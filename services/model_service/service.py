from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from packages.policy_table import PolicyRegistry, PolicyTableTrainer
from packages.solver_like import bucketize_spot
from services.solver_label_service import SolverLabelService


class ModelService:
    def __init__(self, solver_labels: SolverLabelService, model_dir: str = 'var/models') -> None:
        self.solver_labels = solver_labels
        self.registry = PolicyRegistry(model_dir)
        self.trainer = PolicyTableTrainer()

    def train_policy_table(self, *, model_name: str | None = None) -> dict[str, Any]:
        if model_name is None:
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
            model_name = f'policy_table_{stamp}'
        rows = self.solver_labels.build_training_rows_from_spot_packs()
        model = self.trainer.train(rows, model_name=model_name)
        saved = self.registry.save(model)
        saved['training_rows'] = len(rows)
        saved['bucket_count'] = model.metadata['bucket_count']
        saved['model_name'] = model_name
        return saved

    def train_policy_table_from_pokerbench(
        self,
        *,
        max_rows: int = 0,
        equity_samples: int = 0,
        model_name: str | None = None,
        config: str = "easy",
    ) -> dict[str, Any]:
        """Train a PolicyTableModel from HuggingFace PokerBench solver data."""
        if model_name is None:
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
            model_name = f'pokerbench_{config}_{stamp}'

        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError(
                "The 'datasets' library is required for PokerBench training. "
                "Install it with: pip install datasets"
            ) from exc

        from packages.dataset_builder.pokerbench_adapter import PokerBenchAdapter

        ds = load_dataset("RZ412/PokerBench", config, split="train")
        rows = list(ds)
        if max_rows > 0:
            rows = rows[:max_rows]

        adapter = PokerBenchAdapter(equity_samples=equity_samples)
        converted = adapter.convert_batch(rows, config=config)
        if not converted:
            return {'error': 'No rows converted', 'input_rows': len(rows)}

        model = self.trainer.train(converted, model_name=model_name)
        saved = self.registry.save(model)
        saved['training_rows'] = len(converted)
        saved['bucket_count'] = model.metadata['bucket_count']
        saved['model_name'] = model_name
        saved['source'] = 'pokerbench'
        saved['config'] = config
        return saved

    def list_models(self) -> list[dict[str, Any]]:
        return self.registry.list_models()

    def get_model_info(self, model_id: str) -> dict[str, Any]:
        model = self.registry.load(model_id)
        return {
            'model_id': model_id,
            'metadata': model.metadata,
            'bucket_count': model.metadata.get('bucket_count', 0),
        }

    def evaluate_model_on_spot_packs(self, model_id: str) -> dict[str, Any]:
        model = self.registry.load(model_id)
        rows = self.solver_labels.build_training_rows_from_spot_packs()
        results = []
        matches = 0
        action_counts: dict[str, int] = {}
        for row in rows:
            bucket_key = (row.get('bucket_info') or bucketize_spot(row))['bucket_key']
            pred = model.predict(bucket_key, fallback='check')
            aligned = pred['action'] == row['label_action']
            matches += int(aligned)
            action_counts[pred['action']] = action_counts.get(pred['action'], 0) + 1
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
            'action_distribution': action_counts,
            'results': results,
        }

    def compare_models(self, model_id_a: str, model_id_b: str) -> dict[str, Any]:
        """Compare two models side-by-side on spot packs."""
        eval_a = self.evaluate_model_on_spot_packs(model_id_a)
        eval_b = self.evaluate_model_on_spot_packs(model_id_b)
        return {
            'model_a': {'model_id': model_id_a, 'alignment_rate': eval_a['alignment_rate']},
            'model_b': {'model_id': model_id_b, 'alignment_rate': eval_b['alignment_rate']},
            'improvement': round(eval_b['alignment_rate'] - eval_a['alignment_rate'], 4),
        }
