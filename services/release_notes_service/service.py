from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


class ReleaseNotesService:
    def __init__(self, docs_dir: str = 'docs') -> None:
        self.docs_dir = Path(docs_dir)

    def current_notes(self) -> dict:
        sprint_docs = sorted(p.name for p in self.docs_dir.glob('*Sprint*'))
        latest = sprint_docs[-10:]
        return {
            'release_train': 'Sprint 21-30',
            'highlights': [
                'Model governance and model card generation',
                'Release gate and alpha candidate assessment',
                'Batch generation and regression suite manifest',
                'Calibration, deploy manifest and release notes generation',
                'Final handoff documentation pack for local-to-alpha transition',
            ],
            'latest_docs': latest,
        }

    def generate_changelog(self, *, since_tag: str | None = None) -> dict[str, Any]:
        """Generate changelog from git history."""
        commits: list[str] = []
        try:
            cmd = ['git', 'log', '--oneline', '-50']
            if since_tag:
                cmd = ['git', 'log', '--oneline', f'{since_tag}..HEAD']
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=10, check=False
            )
            if result.returncode == 0:
                commits = [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        return {
            'since': since_tag or 'last 50 commits',
            'commit_count': len(commits),
            'commits': commits,
        }
