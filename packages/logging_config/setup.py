from __future__ import annotations

import contextvars
import json
import logging
import os
import sys
from datetime import datetime, timezone

correlation_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "correlation_id", default=""
)


def get_correlation_id() -> str:
    return correlation_id_var.get("")


class JsonFormatter(logging.Formatter):
    """JSON log formatter with correlation ID support."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "correlation_id": get_correlation_id(),
        }
        # Include extra fields
        for key in ("method", "path", "status_code", "duration_ms", "traceback"):
            val = getattr(record, key, None)
            if val is not None:
                log_entry[key] = val
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry, default=str)


def setup_logging(level: str | None = None) -> None:
    """Configure structured JSON logging."""
    log_level = level or os.getenv("POKER_LOG_LEVEL", "INFO")
    root = logging.getLogger()
    root.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Remove existing handlers
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)

    # Quiet noisy loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
