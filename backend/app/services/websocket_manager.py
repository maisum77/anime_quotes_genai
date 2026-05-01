"""
WebSocket Connection Manager
-----------------------------
Manages WebSocket connections for real-time job status updates.
Supports per-job subscriptions, broadcasting, and heartbeat monitoring.
"""

import asyncio
import json
import time
import logging
from typing import Dict, Set, Optional, Any, List
from datetime import datetime, timezone

from fastapi import WebSocket, WebSocketDisconnect

from ..core.config import settings

logger = logging.getLogger(__name__)


class ConnectionInfo:
    """Metadata for an active WebSocket connection."""

    def __init__(self, websocket: WebSocket, user_id: Optional[str] = None):
        self.websocket = websocket
        self.user_id = user_id
        self.subscribed_jobs: Set[str] = set()
        self.subscribed_all: bool = False
        self.connected_at: str = datetime.now(timezone.utc).isoformat()
        self.last_heartbeat: float = time.time()

    @property
    def connection_id(self) -> str:
        return str(id(self.websocket))


class WebSocketManager:
    """
    Manages WebSocket connections for real-time job updates.

    Features:
    - Connection lifecycle management (connect/disconnect)
    - Per-job subscription tracking
    - Global subscription (all jobs)
    - Heartbeat monitoring
    - Broadcast to specific job subscribers or all connections
    """

    def __init__(self) -> None:
        # connection_id → ConnectionInfo
        self._connections: Dict[str, ConnectionInfo] = {}
        # job_id → set of connection_ids
        self._job_subscriptions: Dict[str, Set[str]] = {}
        # Heartbeat task
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._running: bool = False

    @property
    def active_connection_count(self) -> int:
        """Number of active WebSocket connections."""
        return len(self._connections)

    @property
    def total_subscriptions(self) -> int:
        """Total number of job subscriptions across all connections."""
        return sum(len(subs) for subs in self._job_subscriptions.values())

    # ── Lifecycle ────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the heartbeat monitoring task."""
        if not self._running:
            self._running = True
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            logger.info("WebSocketManager started with heartbeat monitoring")

    async def stop(self) -> None:
        """Stop the manager and close all connections."""
        self._running = False
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        # Close all connections
        for conn_id, info in list(self._connections.items()):
            try:
                await info.websocket.close(code=1001, reason="Server shutting down")
            except Exception:
                pass

        self._connections.clear()
        self._job_subscriptions.clear()
        logger.info("WebSocketManager stopped")

    # ── Connection Management ────────────────────────────────────

    async def connect(
        self,
        websocket: WebSocket,
        user_id: Optional[str] = None,
    ) -> str:
        """
        Accept a new WebSocket connection.

        Args:
            websocket: FastAPI WebSocket instance
            user_id: Optional authenticated user ID

        Returns:
            Connection ID string

        Raises:
            HTTPException: If max connections exceeded
        """
        if self.active_connection_count >= settings.ws_max_connections:
            await websocket.close(code=1013, reason="Maximum connections reached")
            raise RuntimeError("Maximum WebSocket connections reached")

        await websocket.accept()

        info = ConnectionInfo(websocket=websocket, user_id=user_id)
        conn_id = info.connection_id
        self._connections[conn_id] = info

        logger.info(
            f"WebSocket connected: {conn_id} "
            f"(user={user_id or 'anonymous'}, "
            f"total={self.active_connection_count})"
        )

        # Send welcome message
        await self._send_to_connection(conn_id, {
            "type": "connected",
            "connection_id": conn_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": "Connected to Anime Quote Generator real-time updates",
        })

        return conn_id

    async def disconnect(self, conn_id: str) -> None:
        """
        Disconnect and clean up a WebSocket connection.

        Args:
            conn_id: Connection ID to disconnect
        """
        info = self._connections.pop(conn_id, None)
        if info:
            # Remove from all job subscriptions
            for job_id in info.subscribed_jobs:
                if job_id in self._job_subscriptions:
                    self._job_subscriptions[job_id].discard(conn_id)
                    if not self._job_subscriptions[job_id]:
                        del self._job_subscriptions[job_id]

            logger.info(
                f"WebSocket disconnected: {conn_id} "
                f"(remaining={self.active_connection_count})"
            )

    # ── Subscriptions ────────────────────────────────────────────

    async def subscribe_to_job(self, conn_id: str, job_id: str) -> None:
        """
        Subscribe a connection to updates for a specific job.

        Args:
            conn_id: Connection ID
            job_id: Job ID to subscribe to
        """
        info = self._connections.get(conn_id)
        if not info:
            return

        info.subscribed_jobs.add(job_id)
        if job_id not in self._job_subscriptions:
            self._job_subscriptions[job_id] = set()
        self._job_subscriptions[job_id].add(conn_id)

        logger.debug(f"Connection {conn_id} subscribed to job {job_id}")

        await self._send_to_connection(conn_id, {
            "type": "subscribed",
            "job_id": job_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    async def subscribe_to_all(self, conn_id: str) -> None:
        """
        Subscribe a connection to all job updates.

        Args:
            conn_id: Connection ID
        """
        info = self._connections.get(conn_id)
        if info:
            info.subscribed_all = True
            await self._send_to_connection(conn_id, {
                "type": "subscribed_all",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

    async def unsubscribe_from_job(self, conn_id: str, job_id: str) -> None:
        """Unsubscribe a connection from a specific job's updates."""
        info = self._connections.get(conn_id)
        if info:
            info.subscribed_jobs.discard(job_id)
        if job_id in self._job_subscriptions:
            self._job_subscriptions[job_id].discard(conn_id)
            if not self._job_subscriptions[job_id]:
                del self._job_subscriptions[job_id]

    # ── Broadcasting ─────────────────────────────────────────────

    async def broadcast_job_update(
        self,
        job_id: str,
        status: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Broadcast a job status update to all subscribed connections.

        Args:
            job_id: Job ID that was updated
            status: New job status
            data: Optional additional data

        Returns:
            Number of connections that received the update
        """
        message = {
            "type": "job_update",
            "job_id": job_id,
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data or {},
        }

        recipient_count = 0

        # Send to job-specific subscribers
        job_subscribers = self._job_subscriptions.get(job_id, set())
        for conn_id in job_subscribers:
            if await self._send_to_connection(conn_id, message):
                recipient_count += 1

        # Send to global subscribers
        for conn_id, info in self._connections.items():
            if info.subscribed_all and conn_id not in job_subscribers:
                if await self._send_to_connection(conn_id, message):
                    recipient_count += 1

        if recipient_count > 0:
            logger.debug(
                f"Broadcast job update: {job_id} → {status} "
                f"({recipient_count} recipients)"
            )

        return recipient_count

    async def broadcast_system_message(
        self,
        message: str,
        level: str = "info",
    ) -> int:
        """
        Broadcast a system message to all connected clients.

        Args:
            message: System message text
            level: Message level (info, warning, error)

        Returns:
            Number of connections that received the message
        """
        payload = {
            "type": "system",
            "level": level,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        recipient_count = 0
        for conn_id in list(self._connections.keys()):
            if await self._send_to_connection(conn_id, payload):
                recipient_count += 1

        return recipient_count

    # ── Private Methods ──────────────────────────────────────────

    async def _send_to_connection(
        self, conn_id: str, data: Dict[str, Any]
    ) -> bool:
        """
        Send data to a specific connection.

        Args:
            conn_id: Target connection ID
            data: Data to send (will be JSON-serialized)

        Returns:
            True if send succeeded, False otherwise
        """
        info = self._connections.get(conn_id)
        if not info:
            return False

        try:
            await info.websocket.send_json(data)
            return True
        except WebSocketDisconnect:
            await self.disconnect(conn_id)
            return False
        except Exception as e:
            logger.warning(f"Failed to send to connection {conn_id}: {e}")
            await self.disconnect(conn_id)
            return False

    async def _heartbeat_loop(self) -> None:
        """Periodically send heartbeat pings and check for stale connections."""
        while self._running:
            try:
                await asyncio.sleep(settings.ws_heartbeat_interval)

                now = time.time()
                stale_connections: List[str] = []

                for conn_id, info in list(self._connections.items()):
                    # Send heartbeat
                    try:
                        await info.websocket.send_json({
                            "type": "heartbeat",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        })
                        info.last_heartbeat = now
                    except Exception:
                        stale_connections.append(conn_id)

                # Clean up stale connections
                for conn_id in stale_connections:
                    await self.disconnect(conn_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat loop error: {e}")

    # ── Status ───────────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        """Get current WebSocket manager status."""
        return {
            "active_connections": self.active_connection_count,
            "total_subscriptions": self.total_subscriptions,
            "subscribed_jobs": list(self._job_subscriptions.keys()),
            "max_connections": settings.ws_max_connections,
        }


# ── Singleton Instance ───────────────────────────────────────────

ws_manager = WebSocketManager()
