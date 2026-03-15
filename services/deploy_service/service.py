from __future__ import annotations

import json
from pathlib import Path


class DeployService:
    def __init__(self, base_dir: str = '.') -> None:
        self.base_dir = Path(base_dir)

    def manifest(self) -> dict:
        env_example = self.base_dir / '.env.example'
        dockerfile = self.base_dir / 'Dockerfile'
        compose = self.base_dir / 'docker-compose.yml'
        configs = self.base_dir / 'configs'
        config_files = sorted(p.name for p in configs.glob('*.json')) if configs.exists() else []
        return {
            'dockerfile_present': dockerfile.exists(),
            'compose_present': compose.exists(),
            'env_example_present': env_example.exists(),
            'config_files': config_files,
            'base_dir': str(self.base_dir.resolve()),
            'deploy_note': 'Packaging assets are present for local/container packaging. Production hardening still requires secrets, observability and runtime controls.',
        }
