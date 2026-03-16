from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any


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

    def validate_package(self) -> dict[str, Any]:
        """Validate deployment package completeness."""
        checks: list[dict[str, Any]] = []
        dockerfile = self.base_dir / 'Dockerfile'
        compose = self.base_dir / 'docker-compose.yml'
        env_example = self.base_dir / '.env.example'

        checks.append({'name': 'Dockerfile', 'present': dockerfile.exists()})
        checks.append({'name': 'docker-compose.yml', 'present': compose.exists()})
        checks.append({'name': '.env.example', 'present': env_example.exists()})

        # Check pyproject.toml
        pyproject = self.base_dir / 'pyproject.toml'
        checks.append({'name': 'pyproject.toml', 'present': pyproject.exists()})

        # Check healthcheck script
        healthcheck = self.base_dir / 'infra' / 'scripts' / 'healthcheck.py'
        checks.append({'name': 'healthcheck.py', 'present': healthcheck.exists()})

        all_pass = all(c['present'] for c in checks)
        return {
            'valid': all_pass,
            'checks': checks,
            'missing': [c['name'] for c in checks if not c['present']],
        }

    def build_info(self) -> dict[str, Any]:
        """Return build metadata from pyproject.toml and git."""
        pyproject = self.base_dir / 'pyproject.toml'
        version = 'unknown'
        if pyproject.exists():
            for line in pyproject.read_text(encoding='utf-8').splitlines():
                if line.strip().startswith('version'):
                    version = line.split('=', 1)[1].strip().strip('"').strip("'")
                    break
        return {
            'version': version,
            'base_dir': str(self.base_dir.resolve()),
            'python_available': shutil.which('python') is not None,
            'docker_available': shutil.which('docker') is not None,
        }
