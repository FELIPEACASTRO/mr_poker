from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

from packages.persistence.database import DatabaseManager


class SagaStore:
    """Persistent saga state backed by SQLite."""

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
                CREATE TABLE IF NOT EXISTS saga_state (
                    saga_id TEXT PRIMARY KEY,
                    saga_type TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'running',
                    current_step INTEGER NOT NULL DEFAULT 0,
                    total_steps INTEGER NOT NULL DEFAULT 0,
                    idempotency_key TEXT UNIQUE,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT,
                    results_json TEXT DEFAULT '[]',
                    error TEXT
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_saga_idempotency ON saga_state(idempotency_key)"
            )

    def create(
        self,
        saga_id: str,
        saga_type: str,
        total_steps: int,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO saga_state (saga_id, saga_type, total_steps, idempotency_key, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                (saga_id, saga_type, total_steps, idempotency_key, now, now),
            )
        return {"saga_id": saga_id, "status": "running"}

    def get(self, saga_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM saga_state WHERE saga_id = ?", (saga_id,)
            ).fetchone()
            if row is None:
                return None
            result = dict(row)
            result["results"] = json.loads(result.pop("results_json"))
            return result

    def get_by_idempotency_key(self, key: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM saga_state WHERE idempotency_key = ?", (key,)
            ).fetchone()
            if row is None:
                return None
            result = dict(row)
            result["results"] = json.loads(result.pop("results_json"))
            return result

    def advance(
        self, saga_id: str, step_index: int, step_result: Any = None
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        existing = self.get(saga_id)
        if existing is None:
            return
        results = existing.get("results", [])
        results.append(step_result)
        with self._connect() as conn:
            conn.execute(
                "UPDATE saga_state SET current_step = ?, results_json = ?, updated_at = ? WHERE saga_id = ?",
                (step_index + 1, json.dumps(results, default=str), now, saga_id),
            )

    def complete(self, saga_id: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "UPDATE saga_state SET status = 'completed', completed_at = ?, updated_at = ? WHERE saga_id = ?",
                (now, now, saga_id),
            )

    def fail(self, saga_id: str, error: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "UPDATE saga_state SET status = 'failed', error = ?, updated_at = ? WHERE saga_id = ?",
                (error, now, saga_id),
            )

    def compensating(self, saga_id: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "UPDATE saga_state SET status = 'compensating', updated_at = ? WHERE saga_id = ?",
                (now, saga_id),
            )
