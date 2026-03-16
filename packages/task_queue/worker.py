from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable
from uuid import uuid4


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TaskInfo:
    task_id: str
    task_type: str
    status: TaskStatus
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    result: Any = None
    error: str | None = None
    progress: float = 0.0


class TaskQueue:
    """Background task execution with status tracking."""

    def __init__(self, max_workers: int = 2) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._tasks: dict[str, TaskInfo] = {}
        self._lock = threading.Lock()

    def submit(
        self,
        task_type: str,
        fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> str:
        """Submit a task for background execution. Returns task_id."""
        task_id = str(uuid4())
        now = datetime.now(timezone.utc).isoformat()
        info = TaskInfo(
            task_id=task_id,
            task_type=task_type,
            status=TaskStatus.PENDING,
            created_at=now,
        )
        with self._lock:
            self._tasks[task_id] = info

        self._executor.submit(self._run_task, task_id, fn, *args, **kwargs)
        return task_id

    def get_task(self, task_id: str) -> TaskInfo | None:
        with self._lock:
            return self._tasks.get(task_id)

    def list_tasks(self, limit: int = 50) -> list[TaskInfo]:
        with self._lock:
            tasks = sorted(
                self._tasks.values(),
                key=lambda t: t.created_at,
                reverse=True,
            )
            return tasks[:limit]

    def _run_task(
        self,
        task_id: str,
        fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            task = self._tasks[task_id]
            task.status = TaskStatus.RUNNING
            task.started_at = now

        try:
            result = fn(*args, **kwargs)
            with self._lock:
                task.status = TaskStatus.COMPLETED
                task.result = result
                task.completed_at = datetime.now(timezone.utc).isoformat()
                task.progress = 1.0
        except Exception as exc:
            with self._lock:
                task.status = TaskStatus.FAILED
                task.error = str(exc)
                task.completed_at = datetime.now(timezone.utc).isoformat()
