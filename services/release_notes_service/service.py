from __future__ import annotations

from pathlib import Path


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
