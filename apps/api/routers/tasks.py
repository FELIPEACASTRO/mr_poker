from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

router = APIRouter(prefix="/v1/tasks")


def _get_task_queue(request: Request):
    task_queue = getattr(request.app.state, "task_queue", None)
    if task_queue is None:
        raise HTTPException(status_code=503, detail="task queue not available")
    return task_queue


@router.get("/{task_id}")
def get_task(task_id: str, request: Request) -> dict:
    queue = _get_task_queue(request)
    task = queue.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="task not found")
    return {
        "task_id": task.task_id,
        "task_type": task.task_type,
        "status": task.status.value,
        "created_at": task.created_at,
        "started_at": task.started_at,
        "completed_at": task.completed_at,
        "result": task.result,
        "error": task.error,
        "progress": task.progress,
    }


@router.get("")
def list_tasks(request: Request, limit: int = 50) -> dict:
    queue = _get_task_queue(request)
    tasks = queue.list_tasks(limit=limit)
    return {
        "tasks": [
            {
                "task_id": t.task_id,
                "task_type": t.task_type,
                "status": t.status.value,
                "created_at": t.created_at,
                "completed_at": t.completed_at,
                "progress": t.progress,
            }
            for t in tasks
        ]
    }
