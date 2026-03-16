from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any

from packages.persistence.database import DatabaseManager

logger = logging.getLogger("audit")


class AuditLogger:
    """Append-only audit trail backed by SQLite."""

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
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    username TEXT NOT NULL,
                    ip TEXT DEFAULT '',
                    resource TEXT DEFAULT '',
                    action TEXT DEFAULT '',
                    details_json TEXT DEFAULT '{}'
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_type ON audit_log(event_type)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_log(username)"
            )

    def _log(
        self,
        event_type: str,
        username: str,
        ip: str = "",
        resource: str = "",
        action: str = "",
        details: dict[str, Any] | None = None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO audit_log (timestamp, event_type, username, ip, resource, action, details_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (now, event_type, username, ip, resource, action, json.dumps(details or {})),
            )
        logger.info(
            "audit: %s user=%s action=%s resource=%s",
            event_type,
            username,
            action,
            resource,
        )

    def log_auth(self, username: str, success: bool, ip: str = "") -> None:
        self._log("auth", username, ip=ip, action="login" if success else "login_failed")

    def log_access(self, username: str, resource: str, action: str) -> None:
        self._log("access", username, resource=resource, action=action)

    def log_admin(self, username: str, action: str, details: dict[str, Any] | None = None) -> None:
        self._log("admin", username, action=action, details=details)

    def get_recent(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            result = []
            for row in rows:
                item = dict(row)
                item["details"] = json.loads(item.pop("details_json"))
                result.append(item)
            return result
