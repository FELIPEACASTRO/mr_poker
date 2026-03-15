from __future__ import annotations

from typing import Any

from packages.persistence import SqliteHandStore
from services.analytics_service import SessionAnalyticsService
from services.opponent_profile_service import OpponentProfileService
from services.taxonomy_service import TaxonomyService


class CoachService:
    def __init__(self, store: SqliteHandStore) -> None:
        self.store = store
        self.analytics = SessionAnalyticsService(store)
        self.profiles = OpponentProfileService(store)
        self.taxonomy = TaxonomyService(store)

    def session_report(self, session_id: str) -> dict[str, Any]:
        analytics = self.analytics.session_analytics(session_id)
        profiles = self.profiles.session_profiles(session_id)
        seat0 = profiles['profiles']['seat0']
        top_taxonomies = sorted(analytics['taxonomy_counts'].items(), key=lambda kv: kv[1], reverse=True)[:5]
        leaks = []
        strengths = []
        if analytics.get('positive_edge_continue_rate', 0.0) < 0.55:
            leaks.append('Continues marginal spots with insufficient edge too often.')
        if analytics.get('avg_edge_by_street', {}).get('river', 0.0) < 0.0:
            leaks.append('River decisions show negative average edge proxy.')
        if seat0.get('style_tag') == 'loose_passive':
            leaks.append('Seat0 profile looks loose-passive; add more pre-flop aggression or tighter calls.')
        if analytics.get('continue_ev_proxy_sum', 0.0) > 0:
            strengths.append('Decision traces show positive local continue EV proxy.')
        if analytics.get('trace_count', 0) >= 20:
            strengths.append('Sufficient trace volume for targeted spot review.')
        study_plan = [
            'Review top negative-edge streets from session analytics.',
            'Replay hands tagged as bluffcatch or priced_fold_disciplined.',
            'Run spot-pack comparisons against solver-like labels before next session.',
        ]
        return {
            'session_id': session_id,
            'summary': {
                'hands': analytics['hand_count'],
                'trace_count': analytics['trace_count'],
                'top_taxonomies': top_taxonomies,
            },
            'strengths': strengths,
            'leaks': leaks,
            'study_plan': study_plan,
            'opponent_profiles': profiles['profiles'],
        }

    def hand_review(self, hand_id: str) -> dict[str, Any]:
        hand = self.store.get_hand(hand_id)
        if hand is None:
            raise KeyError(hand_id)
        traces = self.store.get_decision_traces(hand_id=hand_id)
        taxonomy = self.taxonomy.classify_hand(hand_id)
        review_points = []
        for item in traces:
            trace = item['trace']
            if float(trace.get('estimated_equity', 0.0)) < float(trace.get('pot_odds', 0.0)) and trace.get('action_type') in {'call', 'raise', 'all_in'}:
                review_points.append(f"Seat {trace['actor_seat']} continued below price on {trace['street']}")
        return {
            'hand_id': hand_id,
            'taxonomy': taxonomy,
            'trace_count': len(traces),
            'review_points': review_points,
        }
