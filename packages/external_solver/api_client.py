from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


@dataclass
class SolverAPIConfig:
    """Configuration for external solver API."""

    base_url: str = ""
    api_key: str = ""
    timeout_seconds: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0


class SolverAPIClient:
    """HTTP client for real solver APIs (PioSolver, GTO+, etc.)."""

    def __init__(self, config: SolverAPIConfig | None = None) -> None:
        self.config = config or SolverAPIConfig()
        self._cache: dict[str, Any] = {}

    @property
    def is_configured(self) -> bool:
        return bool(self.config.base_url)

    def solve(
        self,
        *,
        board: list[str],
        hero_range: str,
        villain_range: str,
        pot: int,
        effective_stack: int,
    ) -> dict[str, Any]:
        """Request a solver solution from the external API."""
        if not self.is_configured:
            return {
                "status": "not_configured",
                "message": "external solver API not configured",
                "strategy": {},
            }

        cache_key = f"{board}|{hero_range}|{villain_range}|{pot}|{effective_stack}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        payload = {
            "board": board,
            "hero_range": hero_range,
            "villain_range": villain_range,
            "pot": pot,
            "effective_stack": effective_stack,
        }

        result = self._request("POST", "/solve", payload)
        self._cache[cache_key] = result
        return result

    def get_strategy(self, spot_id: str) -> dict[str, Any]:
        """Retrieve a pre-computed strategy by spot ID."""
        if not self.is_configured:
            return {"status": "not_configured", "strategy": {}}

        cache_key = f"strategy:{spot_id}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        result = self._request("GET", f"/strategies/{spot_id}")
        self._cache[cache_key] = result
        return result

    def _request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make an HTTP request with retry logic."""
        url = f"{self.config.base_url.rstrip('/')}{path}"
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        data = json.dumps(body).encode("utf-8") if body else None

        last_error: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                req = Request(url, data=data, headers=headers, method=method)
                with urlopen(req, timeout=self.config.timeout_seconds) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except (URLError, TimeoutError, OSError) as exc:
                last_error = exc
                logger.warning(
                    "solver API request failed (attempt %d/%d): %s",
                    attempt + 1,
                    self.config.max_retries,
                    exc,
                )
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))

        return {
            "status": "error",
            "message": f"solver API unreachable after {self.config.max_retries} attempts: {last_error}",
            "strategy": {},
        }

    def clear_cache(self) -> int:
        """Clear the solver cache. Returns number of entries cleared."""
        count = len(self._cache)
        self._cache.clear()
        return count
