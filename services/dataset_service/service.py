from __future__ import annotations

from typing import Any

from packages.dataset_builder import DatasetBuilder
from services.solver_label_service import SolverLabelService


class DatasetService:
    def __init__(self, store, solver_labels: SolverLabelService, dataset_dir: str = 'var/datasets') -> None:
        self.store = store
        self.solver_labels = solver_labels
        self.builder = DatasetBuilder(store, report_dir=dataset_dir)

    def build_master_dataset(self, dataset_name: str = 'master_v1') -> dict[str, Any]:
        spot_rows = []
        for pack in self.solver_labels.spot_packs.list_packs():
            comp = self.solver_labels.compare_spot_pack(pack['spot_id'])
            normalized = comp['normalized_spot']
            normalized['source_spot_id'] = pack['spot_id']
            spot_rows.append(normalized)
        rows = self.builder.build_rows(include_spot_packs=spot_rows)
        exported = self.builder.export(rows, dataset_name=dataset_name)
        exported['sample_rows'] = rows[:5]
        return exported

    def list_datasets(self) -> list[dict[str, Any]]:
        return self.builder.list_exports()
