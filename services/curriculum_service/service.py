from __future__ import annotations

from typing import Any

from packages.curriculum import build_curriculum
from services.analytics_service import SessionAnalyticsService
from services.coach_service import CoachService


class CurriculumService:
    def __init__(self, store) -> None:
        self.analytics = SessionAnalyticsService(store)
        self.coach = CoachService(store)

    def build_for_session(self, session_id: str) -> dict:
        analytics = self.analytics.session_analytics(session_id)
        coach_report = self.coach.session_report(session_id)
        return {'session_id': session_id, 'curriculum': build_curriculum(analytics, coach_report)}

    def progression_status(self, session_ids: list[str]) -> dict[str, Any]:
        """Track progression across multiple sessions."""
        progression = []
        for sid in session_ids:
            try:
                analytics = self.analytics.session_analytics(sid)
                progression.append({
                    'session_id': sid,
                    'hand_count': analytics.get('hand_count', 0),
                    'trace_count': analytics.get('trace_count', 0),
                    'avg_edge': analytics.get('avg_edge_by_street', {}),
                })
            except KeyError:
                progression.append({'session_id': sid, 'error': 'session not found'})

        return {
            'sessions_tracked': len(session_ids),
            'progression': progression,
        }
