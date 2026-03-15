from __future__ import annotations

import json
from pathlib import Path

from packages.evaluation import evaluate_policy_model


class EvaluationService:
    def __init__(self, model_registry, dataset_dir: str = 'var/datasets', report_dir: str = 'var/reports') -> None:
        self.model_registry = model_registry
        self.dataset_dir = Path(dataset_dir)
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def _load_rows(self, dataset_id: str, split: str | None = None) -> list[dict]:
        root = self.dataset_dir / dataset_id
        path = root / (f'{split}.jsonl' if split else 'rows.jsonl')
        if not path.exists():
            raise KeyError(dataset_id)
        rows = []
        for line in path.read_text(encoding='utf-8').splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows

    def evaluate_model(self, model_id: str, dataset_id: str, split: str | None = None) -> dict:
        model = self.model_registry.load(model_id)
        rows = self._load_rows(dataset_id, split=split)
        evaluation = evaluate_policy_model(model, rows)
        payload = {
            'model_id': model_id,
            'dataset_id': dataset_id,
            'split': split or 'all',
            'rows': evaluation.rows,
            'accuracy': evaluation.accuracy,
            'split_accuracy': evaluation.split_accuracy,
            'taxonomy_accuracy': evaluation.taxonomy_accuracy,
            'confusion': evaluation.confusion,
            'confidence_bins': evaluation.confidence_bins,
        }
        report_path = self.report_dir / f'evaluation_{model_id}_{dataset_id}_{split or "all"}.json'
        report_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        payload['report_path'] = str(report_path)
        return payload
