from __future__ import annotations

from services.evaluation_service import EvaluationService


class CalibrationService:
    def __init__(self, evaluation: EvaluationService) -> None:
        self.evaluation = evaluation

    def calibrate(self, model_id: str, dataset_id: str, split: str | None = None) -> dict:
        report = self.evaluation.evaluate_model(model_id, dataset_id, split=split)
        bins = report.get('confidence_bins', []) or []
        total = 0
        weighted_gap = 0.0
        for stats in bins:
            count = int(stats.get('count', 0))
            if count <= 0:
                continue
            total += count
            label = str(stats.get('bin', '0.0-1.0'))
            try:
                lo, hi = label.split('-')
                conf = (float(lo) + float(hi)) / 2.0
            except Exception:
                conf = float(stats.get('avg_confidence', 0.0))
            acc = float(stats.get('accuracy', 0.0))
            weighted_gap += abs(conf - acc) * count
        ece = round(weighted_gap / total, 6) if total else 0.0
        return {
            'model_id': model_id,
            'dataset_id': dataset_id,
            'split': split or 'all',
            'rows': report['rows'],
            'ece_proxy': ece,
            'confidence_bins': bins,
            'note': 'ECE proxy derived from local confidence bins.',
        }
