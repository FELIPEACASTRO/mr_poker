from __future__ import annotations

from collections import Counter
from typing import Any


def build_curriculum(analytics: dict[str, Any], coach_report: dict[str, Any]) -> dict[str, Any]:
    taxonomy_counts = analytics.get('taxonomy_counts', {})
    ordered = sorted(taxonomy_counts.items(), key=lambda kv: kv[1], reverse=True)
    drills = [{'focus_area': tag, 'volume': count, 'goal': 'raise decision quality and consistency in this node family'} for tag, count in ordered[:5]]
    priority = Counter()
    if analytics.get('avg_edge_by_street', {}).get('river', 0.0) < 0:
        priority['river'] += 2
    if analytics.get('avg_edge_by_street', {}).get('turn', 0.0) < 0:
        priority['turn'] += 1
    for leak in coach_report.get('leaks', []):
        low = leak.lower()
        if 'pre-flop' in low:
            priority['pre_flop'] += 2
        if 'river' in low:
            priority['river'] += 1
    return {'priorities': [k for k, _ in priority.most_common()], 'drills': drills, 'study_plan': coach_report.get('study_plan', [])}
