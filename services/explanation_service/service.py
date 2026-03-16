from __future__ import annotations

from typing import Any

from packages.persistence import SqliteHandStore


class ExplanationService:
    """Generates human-readable explanations from decision traces and hand state."""

    def __init__(self, store: SqliteHandStore) -> None:
        self.store = store

    def explain_decision(self, trace: dict[str, Any]) -> str:
        action = trace.get("action_type", "unknown")
        equity = trace.get("estimated_equity", 0.0)
        pot_odds = trace.get("pot_odds", 0.0)
        street = trace.get("street", "unknown")
        hole_class = trace.get("hole_class", "unknown")

        parts = [f"**Decision: {action}** on {street}"]

        if equity and pot_odds:
            edge = float(equity) - float(pot_odds)
            parts.append(
                f"Equity: {float(equity):.1%} vs pot odds: {float(pot_odds):.1%} "
                f"(edge: {edge:+.1%})"
            )
            if edge > 0.1:
                parts.append("Strong positive edge supports continuing.")
            elif edge > 0:
                parts.append("Marginal positive edge.")
            else:
                parts.append("Negative edge — discipline required.")

        if hole_class and hole_class != "unknown":
            parts.append(f"Hole cards classified as: {hole_class}")

        board_texture = trace.get("board_texture")
        if board_texture:
            parts.append(f"Board texture: {board_texture}")

        return "\n".join(parts)

    def explain_hand(self, hand_id: str) -> dict[str, Any]:
        hand = self.store.get_hand(hand_id)
        if hand is None:
            raise KeyError(hand_id)

        traces = self.store.get_decision_traces(hand_id=hand_id)
        actions = self.store.get_actions(hand_id)
        latest = self.store.get_latest_snapshot(hand_id)

        explanations = []
        for item in traces:
            trace = item["trace"]
            explanations.append({
                "seat": trace.get("actor_seat"),
                "street": trace.get("street"),
                "action": trace.get("action_type"),
                "explanation": self.explain_decision(trace),
            })

        summary_parts = [f"Hand {hand_id}: {len(actions)} actions across {len(traces)} traced decisions."]
        if latest:
            snap = latest["snapshot"]
            if snap.get("is_terminal"):
                winner = snap.get("winner_seat")
                summary_parts.append(f"Winner: Seat {winner}" if winner is not None else "Split pot")
            else:
                summary_parts.append(f"Hand still in progress on {snap.get('street', 'unknown')}")

        return {
            "hand_id": hand_id,
            "summary": " ".join(summary_parts),
            "decision_explanations": explanations,
            "action_count": len(actions),
            "trace_count": len(traces),
        }
