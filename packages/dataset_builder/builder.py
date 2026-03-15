from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any
from uuid import uuid4

from packages.solver_like import bucketize_spot
from packages.taxonomy.spot_taxonomy import classify_trace


def stable_split(key: str) -> str:
    value = int(hashlib.sha256(key.encode('utf-8')).hexdigest()[:8], 16) % 100
    if value < 70:
        return 'train'
    if value < 85:
        return 'validation'
    return 'test'


class DatasetBuilder:
    def __init__(self, store, report_dir: str = 'var/datasets') -> None:
        self.store = store
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def build_rows(self, include_spot_packs: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for item in self.store.get_decision_traces():
            trace = item['trace']
            bucket_info = bucketize_spot(trace)
            taxonomy_tags = classify_trace(trace)
            source_key = f"trace:{item['hand_id']}:{item['id']}"
            rows.append({
                'row_id': str(uuid4()),
                'source_type': 'trace',
                'source_key': source_key,
                'session_id': item.get('session_id'),
                'hand_id': item['hand_id'],
                'actor_seat': item['actor_seat'],
                'street': trace.get('street'),
                'hole_class': trace.get('hole_class'),
                'board_texture': trace.get('board_texture'),
                'pot': trace.get('pot'),
                'to_call': trace.get('to_call'),
                'pot_odds': trace.get('pot_odds'),
                'estimated_equity': trace.get('estimated_equity'),
                'legal_actions': trace.get('legal_actions', []),
                'bucket_info': bucket_info,
                'label_action': trace.get('action_type'),
                'label_confidence': 1.0,
                'label_source': 'observed_trace',
                'taxonomy_tags': taxonomy_tags,
                'split': stable_split(source_key),
            })

        for spot in include_spot_packs or []:
            solver_like = spot['solver_like']
            bucket_info = spot.get('bucket_info') or solver_like.get('bucket_info') or bucketize_spot(spot)
            pseudo_trace = {
                'street': spot.get('street'),
                'action_type': solver_like.get('label_action'),
                'hole_class': spot.get('hole_class'),
                'board_texture': spot.get('board_texture'),
                'to_call': spot.get('to_call', 0),
                'estimated_equity': spot.get('estimated_equity', 0.0),
                'pot_odds': spot.get('pot_odds', 0.0),
            }
            taxonomy_tags = classify_trace(pseudo_trace)
            source_key = f"spot_pack:{spot['source_spot_id']}"
            rows.append({
                'row_id': str(uuid4()),
                'source_type': 'spot_pack',
                'source_key': source_key,
                'session_id': None,
                'hand_id': spot.get('hand_id'),
                'actor_seat': spot.get('actor_seat'),
                'street': spot.get('street'),
                'hole_class': spot.get('hole_class'),
                'board_texture': spot.get('board_texture'),
                'pot': spot.get('pot'),
                'to_call': spot.get('to_call'),
                'pot_odds': spot.get('pot_odds'),
                'estimated_equity': spot.get('estimated_equity'),
                'legal_actions': spot.get('legal_actions', []),
                'bucket_info': bucket_info,
                'label_action': solver_like.get('label_action'),
                'label_confidence': solver_like.get('confidence', 0.0),
                'label_source': 'solver_like_spot_pack',
                'taxonomy_tags': taxonomy_tags,
                'split': stable_split(source_key),
                'source_spot_id': spot['source_spot_id'],
            })
        return rows

    def build_manifest(self, rows: list[dict[str, Any]], *, dataset_name: str = 'master_v1') -> dict[str, Any]:
        split_counts = Counter(row['split'] for row in rows)
        source_counts = Counter(row['source_type'] for row in rows)
        taxonomy_counts = Counter(tag for row in rows for tag in row.get('taxonomy_tags', []))
        return {
            'dataset_name': dataset_name,
            'row_count': len(rows),
            'split_counts': dict(split_counts),
            'source_counts': dict(source_counts),
            'taxonomy_counts': dict(taxonomy_counts),
        }

    def export(self, rows: list[dict[str, Any]], *, dataset_name: str = 'master_v1') -> dict[str, Any]:
        manifest = self.build_manifest(rows, dataset_name=dataset_name)
        dataset_id = str(uuid4())
        target = self.report_dir / dataset_id
        target.mkdir(parents=True, exist_ok=True)

        rows_payload = "\n".join(json.dumps(r) for r in rows) + ("\n" if rows else "")
        (target / 'rows.jsonl').write_text(rows_payload, encoding='utf-8')
        payload = {'dataset_id': dataset_id, **manifest}
        (target / 'manifest.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')

        for split in ('train', 'validation', 'test'):
            split_rows = [r for r in rows if r['split'] == split]
            split_payload = "\n".join(json.dumps(r) for r in split_rows) + ("\n" if split_rows else "")
            (target / f'{split}.jsonl').write_text(split_payload, encoding='utf-8')

        return {
            **payload,
            'path': str(target),
            'jsonl_path': str(target / 'rows.jsonl'),
            'manifest_path': str(target / 'manifest.json'),
        }

    def list_exports(self) -> list[dict[str, Any]]:
        return [json.loads(p.read_text(encoding='utf-8')) for p in sorted(self.report_dir.glob('*/manifest.json'))]
