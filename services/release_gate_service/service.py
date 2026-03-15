from __future__ import annotations

from services.readiness_service import ReadinessService
from services.evaluation_service import EvaluationService


class ReleaseGateService:
    def __init__(self, readiness: ReadinessService, evaluation: EvaluationService) -> None:
        self.readiness = readiness
        self.evaluation = evaluation

    def evaluate_candidate(self, model_id: str, dataset_id: str, *, min_accuracy: float = 0.55, min_rows: int = 10) -> dict:
        readiness = self.readiness.snapshot()
        report = self.evaluation.evaluate_model(model_id, dataset_id)
        blockers = list(readiness.get('blockers', []))
        if report['rows'] < min_rows:
            blockers.append(f'Dataset rows below minimum threshold: {report["rows"]} < {min_rows}.')
        if report['accuracy'] < min_accuracy:
            blockers.append(f'Accuracy below threshold: {report["accuracy"]:.4f} < {min_accuracy:.4f}.')
        return {
            'model_id': model_id,
            'dataset_id': dataset_id,
            'readiness': readiness,
            'evaluation': report,
            'thresholds': {'min_accuracy': min_accuracy, 'min_rows': min_rows},
            'go_for_alpha': len(blockers) == 0,
            'blockers': blockers,
        }
