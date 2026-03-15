from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class GovernanceService:
    def __init__(self, model_dir: str = 'var/models', report_dir: str = 'var/reports', output_dir: str = 'var/model_cards') -> None:
        self.model_dir = Path(model_dir)
        self.report_dir = Path(report_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _latest_evaluations(self) -> dict[str, dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        for path in sorted(self.report_dir.glob('evaluation_*.json')):
            payload = json.loads(path.read_text(encoding='utf-8'))
            model_id = payload.get('model_id')
            if model_id:
                latest[model_id] = payload
        return latest

    def build_model_card(self, model_id: str) -> dict[str, Any]:
        model_path = self.model_dir / f'{model_id}.json'
        if not model_path.exists():
            raise FileNotFoundError(model_id)
        model_payload = json.loads(model_path.read_text(encoding='utf-8'))
        evaluations = self._latest_evaluations()
        latest_eval = evaluations.get(model_id)
        card = {
            'model_id': model_payload['model_id'],
            'metadata': model_payload.get('metadata', {}),
            'training_artifact_path': str(model_path),
            'latest_evaluation': latest_eval,
            'intended_use': 'local-first heads-up NLHE baseline/adaptive research workflow',
            'limitations': [
                'Not a real-time external solver.',
                'Current policy table is a lightweight local model, not a production strategic core.',
                'Evaluation metrics are local proxies and dataset-scoped.'
            ],
            'safety_scope': 'For local sandbox, coaching, replay and research workflows only.',
        }
        out = self.output_dir / f'{model_id}.card.json'
        out.write_text(json.dumps(card, indent=2), encoding='utf-8')
        card['path'] = str(out)
        return card

    def list_model_cards(self) -> list[dict[str, Any]]:
        cards = []
        for path in sorted(self.output_dir.glob('*.card.json')):
            payload = json.loads(path.read_text(encoding='utf-8'))
            payload['path'] = str(path)
            cards.append(payload)
        return cards
