from __future__ import annotations

import os
import shutil
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
        asset_readiness = len(blockers) == 0

        operational_blockers = []
        if shutil.which('ruff') is None:
            operational_blockers.append('ruff executable not found in PATH.')

        makefile = self.base_dir / 'Makefile'
        if makefile.exists():
            make_text = makefile.read_text(encoding='utf-8')
            if 'bash' in make_text and os.name == 'nt':
                operational_blockers.append('Makefile validation targets depend on bash in Windows environment.')

        external_solver_sample = self.base_dir / 'var' / 'external_solver' / 'sample_solver_v1.json'
        if external_solver_sample.exists():
            operational_blockers.append('External solver integration is in sample/mock local mode.')

        hardening_markers = [
            self.base_dir / 'infra' / 'k8s',
            self.base_dir / 'infra' / 'terraform',
            self.base_dir / 'infra' / 'observability',
        ]
        if not any(marker.exists() for marker in hardening_markers):
            operational_blockers.append('Operational hardening markers not found (k8s/terraform/observability).')

        operational_readiness = asset_readiness and len(operational_blockers) == 0
        deprecations = ['production_candidate is deprecated; use operational_readiness instead.']

        return {
            'docs_ready': docs.exists(),
            'tests_ready': tests.exists(),
            'dataset_count': len(datasets),
            'model_count': len(models),
            'docker_assets_ready': dockerfile.exists(),
            'production_candidate': operational_readiness,
            'blockers': blockers,
            'asset_readiness': asset_readiness,
            'operational_readiness': operational_readiness,
            'operational_blockers': operational_blockers,
            'deprecations': deprecations,
        }
