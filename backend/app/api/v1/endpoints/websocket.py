"""
WebSocket Endpoint
------------------
Real-time job status updates via WebSocket connections.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query

from ....core.security import decode_token
from ....core.config import settings
from ....models.schemas import WSSubscription, WSJobUpdate, WSHeartbeat
from ....services.websocket_manager import ws_manager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["WebSocket"])


@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    token: Optional[str] = Query(default=None),
) -> None:
    """
    WebSocket endpoint for real-time job status updates.

    Connection:
        Connect to `ws://host:port/api/v1/ws?token=<jwt_token>`

    Authentication:
        Pass a valid JWT access token as the `token` query parameter.
        If no token is provided, the connection is accepted as anonymous.

    Messages:
        - Client → Server: `{"action": "subscribe", "job_ids": ["job-123", ...]}`
        - Client → Server: `{"action": "subscribe_all"}`
        - Server → Client: `{"type": "job_update", "job_id": "...", "status": "...", ...}`
        - Server → Client: `{"type": "heartbeat", "timestamp": "..."}`
        - Server → Client: `{"type": "system", "level": "info", "message": "..."}`

    Heartbeat:
        The server sends heartbeat messages every 30 seconds.
        If the client fails to respond, the connection is closed.
    """
    # Authenticate (optional)
    user_id: Optional[str] = None
    if token:
        try:
            token_data = decode_token(token)
            user_id = token_data.user_id
        except Exception:
            await websocket.close(code=4001, reason="Invalid authentication token")
            return

    # Connect
    conn_id = await ws_manager.connect(websocket, user_id=user_id)

    try:
        while True:
            # Receive messages from client
            raw_data = await websocket.receive_text()

            try:
                data = json.loads(raw_data)
                action = data.get("action", "")

                if action == "subscribe":
                    # Subscribe to specific job updates
                    job_ids = data.get("job_ids", [])
                    for job_id in job_ids:
                        await ws_manager.subscribe_to_job(conn_id, job_id)

                elif action == "subscribe_all":
                    # Subscribe to all job updates
                    await ws_manager.subscribe_to_all(conn_id)

                elif action == "unsubscribe":
                    # Unsubscribe from specific jobs
                    job_ids = data.get("job_ids", [])
                    for job_id in job_ids:
                        await ws_manager.unsubscribe_from_job(conn_id, job_id)

                elif action == "ping":
                    # Respond to client ping
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })

                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Unknown action: {action}",
                        "valid_actions": ["subscribe", "subscribe_all", "unsubscribe", "ping"],
                    })

            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON message",
                })

    except WebSocketDisconnect:
        await ws_manager.disconnect(conn_id)
        logger.info(f"WebSocket client disconnected: {conn_id}")
    except Exception as e:
        logger.error(f"WebSocket error for {conn_id}: {e}")
        await ws_manager.disconnect(conn_id)


@router.get(
    "/ws/status",
    summary="WebSocket status",
    description="Get the current status of the WebSocket manager.",
)
async def websocket_status() -> dict:
    """Get WebSocket connection statistics."""
    return ws_manager.get_status()
