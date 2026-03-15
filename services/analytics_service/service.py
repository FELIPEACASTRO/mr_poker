from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean

from packages.persistence import SqliteHandStore
from packages.taxonomy import classify_hand, classify_trace


class SessionAnalyticsService:
    def __init__(self, store: SqliteHandStore) -> None:
        self.store = store

    def session_analytics(self, session_id: str) -> dict:
        session = self.store.get_session(session_id)
        if session is None:
            raise KeyError(session_id)
        hands = self.store.get_hands_for_session(session_id)
        traces = self.store.get_decision_traces(session_id=session_id)

        street_counts = Counter()
        action_counts = Counter()
        taxonomy_counts = Counter()
        board_texture_counts = Counter()
        hole_class_counts = Counter()
        edge_by_street: dict[str, list[float]] = defaultdict(list)
        seat_edge: dict[int, list[float]] = defaultdict(list)
        hand_taxonomy = []

        for trace_row in traces:
            trace = trace_row['trace']
            street = str(trace.get('street', 'unknown'))
            street_counts[street] += 1
            action_counts[str(trace.get('action_type', 'unknown'))] += 1
            board_texture_counts[str(trace.get('board_texture', 'unknown'))] += 1
            hole_class_counts[str(trace.get('hole_class', 'unknown'))] += 1
            edge = float(trace.get('estimated_equity', 0.0) or 0.0) - float(trace.get('pot_odds', 0.0) or 0.0)
            edge_by_street[street].append(edge)
            seat_edge[int(trace.get('actor_seat', 0))].append(edge)
            for tag in classify_trace(trace):
                taxonomy_counts[tag] += 1

        for hand in hands:
            latest = self.store.get_latest_snapshot(hand['hand_id'])
            if latest is None:
                continue
            hand_traces = [row['trace'] for row in self.store.get_decision_traces(hand_id=hand['hand_id'])]
            hand_actions = self.store.get_actions(hand['hand_id'])
            hand_classification = classify_hand(latest['snapshot'], hand_traces, hand_actions)
            hand_taxonomy.append(hand_classification)
            for tag in hand_classification['taxonomy_tags']:
                taxonomy_counts[f'hand:{tag}'] += 1

        analytics = {
            'session_id': session_id,
            'hand_count': len(hands),
            'trace_count': len(traces),
            'street_trace_counts': dict(street_counts),
            'action_type_counts': dict(action_counts),
            'board_texture_counts': dict(board_texture_counts),
            'top_hole_classes': hole_class_counts.most_common(10),
            'taxonomy_counts': dict(taxonomy_counts),
            'avg_edge_by_street': {street: round(mean(values), 4) for street, values in edge_by_street.items()},
            'avg_edge_by_seat': {str(seat): round(mean(values), 4) for seat, values in seat_edge.items()},
            'hand_taxonomy': hand_taxonomy,
            'analytics_note': 'Local analytics are derived from baseline-agent traces and canonical persisted state. They are useful for diagnostics, not a substitute for exploitability or true solver EV.',
        }
        return analytics
