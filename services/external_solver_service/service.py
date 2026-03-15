from __future__ import annotations

from packages.external_solver import ExternalSolverAdapter
from services.solver_label_service import SolverLabelService


class ExternalSolverService:
    def __init__(self, solver_labels: SolverLabelService, base_dir: str = 'var/external_solver') -> None:
        self.solver_labels = solver_labels
        self.adapter = ExternalSolverAdapter(base_dir=base_dir)

    def catalog(self) -> dict:
        return {'catalog': self.adapter.catalog()}

    def compare_spot_pack(self, name: str, spot_id: str) -> dict:
        comparison = self.solver_labels.compare_spot_pack(spot_id)
        normalized = comparison['normalized_spot']
        bucket_key = normalized['bucket_info']['bucket_key']
        external = self.adapter.compare(name, spot_id=spot_id, bucket_key=bucket_key)
        return {'spot_id': spot_id, 'solver_name': name, 'bucket_key': bucket_key, 'baseline_action': comparison['baseline_action'], 'solver_like_action': comparison['solver_like_action'], 'external_solver': external}
