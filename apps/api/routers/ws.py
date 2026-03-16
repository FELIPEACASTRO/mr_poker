from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

router = APIRouter()


class ConnectionManager:
    """Manages WebSocket connections for real-time hand state streaming."""

    def __init__(self) -> None:
        self.active_connections: dict[str, list[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, hand_id: str) -> None:
        await websocket.accept()
        if hand_id not in self.active_connections:
            self.active_connections[hand_id] = []
        self.active_connections[hand_id].append(websocket)

    def disconnect(self, websocket: WebSocket, hand_id: str) -> None:
        if hand_id in self.active_connections:
            self.active_connections[hand_id] = [
                ws for ws in self.active_connections[hand_id] if ws != websocket
            ]
            if not self.active_connections[hand_id]:
                del self.active_connections[hand_id]

    async def broadcast(self, hand_id: str, data: dict[str, Any]) -> None:
        if hand_id not in self.active_connections:
            return
        message = json.dumps(data, default=str)
        disconnected = []
        for ws in self.active_connections[hand_id]:
            try:
                await ws.send_text(message)
            except Exception as exc:
                logger.warning("WebSocket send failed for hand %s: %s", hand_id, exc)
                disconnected.append(ws)
        for ws in disconnected:
            self.disconnect(ws, hand_id)


manager = ConnectionManager()


@router.websocket("/ws/hands/{hand_id}")
async def hand_websocket(websocket: WebSocket, hand_id: str) -> None:
    await manager.connect(websocket, hand_id)
    try:
        while True:
            await websocket.receive_text()
            # Echo back any received messages (for ping/pong)
            await websocket.send_text(json.dumps({"type": "ack", "hand_id": hand_id}))
    except WebSocketDisconnect:
        logger.debug("WebSocket disconnected for hand %s", hand_id)
        manager.disconnect(websocket, hand_id)
    except Exception as exc:
        logger.warning("WebSocket error for hand %s: %s", hand_id, exc)
        manager.disconnect(websocket, hand_id)
