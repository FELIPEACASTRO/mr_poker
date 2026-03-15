from __future__ import annotations

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
