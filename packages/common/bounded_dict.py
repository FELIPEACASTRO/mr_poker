from __future__ import annotations

import threading
import time
from typing import TypeVar

K = TypeVar("K")
V = TypeVar("V")


class BoundedDict:
    """Thread-safe dictionary with max size and TTL eviction."""

    def __init__(self, max_size: int = 1000, ttl_seconds: float = 3600) -> None:
        self._data: dict[str, tuple[object, float]] = {}
        self._lock = threading.Lock()
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds

    def __setitem__(self, key: str, value: object) -> None:
        with self._lock:
            self._evict_expired()
            if len(self._data) >= self.max_size and key not in self._data:
                # Evict oldest entry
                oldest_key = min(self._data, key=lambda k: self._data[k][1])
                del self._data[oldest_key]
            self._data[key] = (value, time.monotonic())

    def __getitem__(self, key: str) -> object:
        with self._lock:
            if key not in self._data:
                raise KeyError(key)
            value, ts = self._data[key]
            if time.monotonic() - ts > self.ttl_seconds:
                del self._data[key]
                raise KeyError(key)
            return value

    def __contains__(self, key: str) -> bool:
        with self._lock:
            if key not in self._data:
                return False
            _, ts = self._data[key]
            if time.monotonic() - ts > self.ttl_seconds:
                del self._data[key]
                return False
            return True

    def __delitem__(self, key: str) -> None:
        with self._lock:
            del self._data[key]

    def get(self, key: str, default: object = None) -> object:
        try:
            return self[key]
        except KeyError:
            return default

    def __len__(self) -> int:
        with self._lock:
            self._evict_expired()
            return len(self._data)

    def _evict_expired(self) -> None:
        now = time.monotonic()
        expired = [k for k, (_, ts) in self._data.items() if now - ts > self.ttl_seconds]
        for k in expired:
            del self._data[k]
