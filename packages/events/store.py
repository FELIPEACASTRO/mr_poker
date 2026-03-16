from __future__ import annotations

import json
import sqlite3
from typing import Any

from packages.events.models import DomainEvent
from packages.persistence.database import DatabaseManager


class EventStore:
    """Append-only event store backed by SQLite."""

    def __init__(self, db: DatabaseManager | None = None, db_path: str = "var/poker_ai_local.db") -> None:
        if db is not None:
            self._db = db
        else:
            self._db = DatabaseManager(db_path)
        self._init_table()

    def _connect(self) -> sqlite3.Connection:
        return self._db.connect()

    def _init_table(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS domain_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    event_type TEXT NOT NULL,
                    aggregate_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_aggregate ON domain_events(aggregate_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_type ON domain_events(event_type)"
            )

    def append(self, event: DomainEvent) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO domain_events (event_id, event_type, aggregate_id, payload_json, timestamp) VALUES (?, ?, ?, ?, ?)",
                (
                    event.event_id,
                    event.event_type,
                    event.aggregate_id,
                    json.dumps(event.payload),
                    event.timestamp,
                ),
            )

    def get_events(
        self,
        aggregate_id: str | None = None,
        event_type: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM domain_events"
        params: list[Any] = []
        clauses: list[str] = []
        if aggregate_id is not None:
            clauses.append("aggregate_id = ?")
            params.append(aggregate_id)
        if event_type is not None:
            clauses.append("event_type = ?")
            params.append(event_type)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY id LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
            results = []
            for row in rows:
                item = dict(row)
                item["payload"] = json.loads(item.pop("payload_json"))
                results.append(item)
            return results

    def replay(self, aggregate_id: str) -> list[DomainEvent]:
        """Replay all events for a given aggregate."""
        raw = self.get_events(aggregate_id=aggregate_id)
        return [
            DomainEvent(
                event_id=r["event_id"],
                event_type=r["event_type"],
                aggregate_id=r["aggregate_id"],
                payload=r["payload"],
                timestamp=r["timestamp"],
            )
            for r in raw
        ]
