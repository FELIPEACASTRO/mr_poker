from __future__ import annotations

from pathlib import Path


class ReadinessService:
    def __init__(self, base_dir: str = '.', dataset_dir: str = 'var/datasets', model_dir: str = 'var/models') -> None:
        self.base_dir = Path(base_dir)
        self.dataset_dir = self.base_dir / dataset_dir
        self.model_dir = self.base_dir / model_dir

    def snapshot(self) -> dict:
        docs = self.base_dir / 'docs'
        tests = self.base_dir / 'tests'
        dockerfile = self.base_dir / 'Dockerfile'
        datasets = list(self.dataset_dir.glob('*/manifest.json')) if self.dataset_dir.exists() else []
        models = list(self.model_dir.glob('*.json')) if self.model_dir.exists() else []
        blockers = []
        if not datasets:
            blockers.append('No exported dataset manifests found.')
        if not models:
            blockers.append('No trained policy models found.')
        if not dockerfile.exists():
            blockers.append('Packaging assets missing (Dockerfile not found).')
        return {'docs_ready': docs.exists(), 'tests_ready': tests.exists(), 'dataset_count': len(datasets), 'model_count': len(models), 'docker_assets_ready': dockerfile.exists(), 'production_candidate': len(blockers) == 0, 'blockers': blockers}
