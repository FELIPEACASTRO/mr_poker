from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Application configuration loaded from environment + config files."""

    db_path: str = "var/poker_ai_local.db"
    auth_enabled: bool = False
    jwt_secret: str = "dev-secret-change-me"
    cors_origins: list[str] = field(
        default_factory=lambda: ["http://localhost:5173", "http://localhost:8000"]
    )
    rate_limit_rpm: int = 120
    log_level: str = "INFO"
    otel_enabled: bool = False
    otel_endpoint: str = "http://localhost:4317"
    env: str = "local"
    base_dir: str = "."
    session_max_workers: int = 4
    session_parallel_threshold: int = 9999

    @classmethod
    def from_env(cls) -> Settings:
        """Load settings from environment variables and optional config file."""
        env = os.getenv("POKER_AI_ENV", "local")
        config_file = Path(f"configs/app.{env}.json")
        file_config: dict = {}
        if config_file.exists():
            file_config = json.loads(config_file.read_text())

        cors_raw = os.getenv("POKER_CORS_ORIGINS", file_config.get("cors_origins", "*"))
        if isinstance(cors_raw, str):
            cors_origins = [o.strip() for o in cors_raw.split(",") if o.strip()]
        else:
            cors_origins = list(cors_raw)

        return cls(
            db_path=os.getenv("POKER_AI_DB", file_config.get("db_path", "var/poker_ai_local.db")),
            auth_enabled=os.getenv(
                "POKER_AUTH_ENABLED", str(file_config.get("auth_enabled", "false"))
            ).lower() == "true",
            jwt_secret=os.getenv(
                "POKER_JWT_SECRET", file_config.get("jwt_secret", "dev-secret-change-me")
            ),
            cors_origins=cors_origins or ["*"],
            rate_limit_rpm=int(
                os.getenv("POKER_RATE_LIMIT_RPM", str(file_config.get("rate_limit_rpm", 120)))
            ),
            log_level=os.getenv("POKER_LOG_LEVEL", file_config.get("log_level", "INFO")),
            otel_enabled=os.getenv(
                "POKER_OTEL_ENABLED", str(file_config.get("otel_enabled", "false"))
            ).lower() == "true",
            otel_endpoint=os.getenv(
                "POKER_OTEL_ENDPOINT",
                file_config.get("otel_endpoint", "http://localhost:4317"),
            ),
            env=env,
            base_dir=os.getenv("POKER_BASE_DIR", file_config.get("base_dir", ".")),
            session_max_workers=int(
                os.getenv(
                    "POKER_SESSION_MAX_WORKERS",
                    str(file_config.get("session_max_workers", 4)),
                )
            ),
            session_parallel_threshold=int(
                os.getenv(
                    "POKER_SESSION_PARALLEL_THRESHOLD",
                    str(file_config.get("session_parallel_threshold", 9999)),
                )
            ),
        )
