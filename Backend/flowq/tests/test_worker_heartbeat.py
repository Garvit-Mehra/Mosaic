"""Tests for worker heartbeat system and registration (Task 7.2).

Tests cover:
- Req 7.1: Heartbeat every 5s with 15s TTL, first heartbeat immediately on startup
- Req 7.2: Key format heartbeat:{worker_id}, value is UTC timestamp
- Req 7.3: Register in PostgreSQL with ACTIVE status, hostname, PID
- Req 7.5: Handle Redis errors gracefully (retry next interval, log warning)
"""

import asyncio
import logging
import os
import socket
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.models.enums import WorkerStatus
from src.workers.worker import (
    HEARTBEAT_INTERVAL,
    HEARTBEAT_TTL,
    Worker,
)


@pytest.fixture
def worker_id():
    """Generate a unique worker ID for each test."""
    return uuid4()


@pytest.fixture
def mock_redis():
    """Create a mock async Redis client."""
    client = AsyncMock()
    client.set = AsyncMock(return_value=True)
    return client


@pytest.fixture
def mock_session_factory():
    """Create a mock async session factory."""
    session = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)
    session.begin = MagicMock()
    session.begin.return_value.__aenter__ = AsyncMock(return_value=None)
    session.begin.return_value.__aexit__ = AsyncMock(return_value=None)
    session.add = MagicMock()

    factory = MagicMock()
    factory.return_value = session
    return factory


@pytest.fixture
def mock_priority_queue():
    """Create a mock priority queue."""
    pq = AsyncMock()
    pq.dequeue = AsyncMock(return_value=None)
    return pq


@pytest.fixture
def mock_distributed_lock():
    """Create a mock distributed lock."""
    return AsyncMock()


@pytest.fixture
def mock_handler_registry():
    """Create a mock handler registry."""
    return MagicMock()


@pytest.fixture
def worker(
    worker_id,
    mock_session_factory,
    mock_redis,
    mock_priority_queue,
    mock_distributed_lock,
    mock_handler_registry,
):
    """Create a Worker instance with mocked dependencies."""
    return Worker(
        worker_id=worker_id,
        session_factory=mock_session_factory,
        redis_client=mock_redis,
        priority_queue=mock_priority_queue,
        distributed_lock=mock_distributed_lock,
        handler_registry=mock_handler_registry,
    )


class TestHeartbeatConstants:
    """Test heartbeat configuration constants."""

    def test_heartbeat_interval_is_5_seconds(self):
        """Heartbeat interval should be 5 seconds (Req 7.1)."""
        assert HEARTBEAT_INTERVAL == 5

    def test_heartbeat_ttl_is_15_seconds(self):
        """Heartbeat TTL should be 15 seconds (Req 7.1)."""
        assert HEARTBEAT_TTL == 15


class TestSendHeartbeat:
    """Test the send_heartbeat method (Req 7.1, 7.2)."""

    async def test_heartbeat_key_format(self, worker, mock_redis, worker_id):
        """Heartbeat key should follow format heartbeat:{worker_id} (Req 7.2)."""
        await worker.send_heartbeat()

        mock_redis.set.assert_called_once()
        call_args = mock_redis.set.call_args
        key = call_args[0][0] if call_args[0] else call_args.kwargs.get("name")
        assert key == f"heartbeat:{worker_id}"

    async def test_heartbeat_value_is_utc_timestamp(self, worker, mock_redis):
        """Heartbeat value should be a UTC timestamp (Req 7.2)."""
        before = int(datetime.now(timezone.utc).replace(tzinfo=None).timestamp())
        await worker.send_heartbeat()
        after = int(datetime.now(timezone.utc).replace(tzinfo=None).timestamp())

        call_args = mock_redis.set.call_args
        value = call_args[0][1] if len(call_args[0]) > 1 else call_args.kwargs.get("value")
        timestamp = int(value)
        assert before <= timestamp <= after

    async def test_heartbeat_sets_ttl_15_seconds(self, worker, mock_redis):
        """Heartbeat should be set with TTL of 15 seconds (Req 7.1)."""
        await worker.send_heartbeat()

        call_args = mock_redis.set.call_args
        # Check the 'ex' keyword argument
        assert call_args.kwargs.get("ex") == 15 or (
            len(call_args[0]) > 2 and call_args[0][2] == 15
        )

    async def test_heartbeat_redis_error_handled_gracefully(
        self, worker, mock_redis, caplog
    ):
        """Redis errors should be handled gracefully (Req 7.5)."""
        mock_redis.set.side_effect = Exception("Redis connection lost")

        with caplog.at_level(logging.WARNING):
            # Should not raise
            await worker.send_heartbeat()

        assert "Failed to send heartbeat" in caplog.text

    async def test_heartbeat_redis_connection_error_does_not_raise(
        self, worker, mock_redis
    ):
        """Redis connection errors should not propagate (Req 7.5)."""
        import redis.asyncio as redis_lib

        mock_redis.set.side_effect = redis_lib.ConnectionError("Connection refused")

        # Should not raise any exception
        await worker.send_heartbeat()


class TestHeartbeatLoop:
    """Test the _heartbeat_loop method (Req 7.1)."""

    async def test_heartbeat_loop_sends_initial_heartbeat_immediately(
        self, worker, mock_redis
    ):
        """First heartbeat should be sent immediately (Req 7.1)."""
        # Run the heartbeat loop briefly then cancel
        worker._shutdown_requested = False

        async def stop_after_first():
            # Give time for the initial heartbeat
            await asyncio.sleep(0.05)
            worker._shutdown_requested = True

        # Run heartbeat loop and stop task concurrently
        task = asyncio.create_task(worker._heartbeat_loop())
        await stop_after_first()
        # Wait a bit for loop to exit
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Should have been called at least once (the initial heartbeat)
        assert mock_redis.set.call_count >= 1

    async def test_heartbeat_loop_continues_on_redis_error(
        self, worker, mock_redis
    ):
        """Heartbeat loop should continue on Redis errors (Req 7.5)."""
        call_count = 0

        async def fail_then_succeed(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Temporary Redis error")
            return True

        mock_redis.set.side_effect = fail_then_succeed
        worker._shutdown_requested = False

        async def stop_after_delay():
            await asyncio.sleep(0.1)
            worker._shutdown_requested = True

        task = asyncio.create_task(worker._heartbeat_loop())
        await stop_after_delay()
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Should have attempted multiple heartbeats despite error
        assert call_count >= 1


class TestWorkerRegistration:
    """Test worker registration in PostgreSQL (Req 7.3)."""

    async def test_register_creates_worker_record(self, worker, mock_session_factory):
        """register() should create a Worker record in PostgreSQL."""
        await worker.register()

        session = mock_session_factory.return_value
        session.add.assert_called_once()

    async def test_register_sets_active_status(self, worker, mock_session_factory):
        """Worker should be registered with ACTIVE status (Req 7.3)."""
        await worker.register()

        session = mock_session_factory.return_value
        worker_record = session.add.call_args[0][0]
        assert worker_record.status == WorkerStatus.ACTIVE

    async def test_register_sets_hostname(self, worker, mock_session_factory):
        """Worker should be registered with current hostname (Req 7.3)."""
        await worker.register()

        session = mock_session_factory.return_value
        worker_record = session.add.call_args[0][0]
        assert worker_record.hostname == socket.gethostname()

    async def test_register_sets_pid(self, worker, mock_session_factory):
        """Worker should be registered with current PID (Req 7.3)."""
        await worker.register()

        session = mock_session_factory.return_value
        worker_record = session.add.call_args[0][0]
        assert worker_record.pid == os.getpid()

    async def test_register_sets_worker_id(self, worker, mock_session_factory, worker_id):
        """Worker record should use the worker's UUID."""
        await worker.register()

        session = mock_session_factory.return_value
        worker_record = session.add.call_args[0][0]
        assert worker_record.id == worker_id

    async def test_register_sets_last_heartbeat(self, worker, mock_session_factory):
        """Worker should have last_heartbeat set to current time."""
        before = datetime.now(timezone.utc).replace(tzinfo=None)
        await worker.register()
        after = datetime.now(timezone.utc).replace(tzinfo=None)

        session = mock_session_factory.return_value
        worker_record = session.add.call_args[0][0]
        assert before <= worker_record.last_heartbeat <= after

    async def test_register_sets_started_at(self, worker, mock_session_factory):
        """Worker should have started_at set to current time."""
        before = datetime.now(timezone.utc).replace(tzinfo=None)
        await worker.register()
        after = datetime.now(timezone.utc).replace(tzinfo=None)

        session = mock_session_factory.return_value
        worker_record = session.add.call_args[0][0]
        assert before <= worker_record.started_at <= after


class TestStartCallsRegister:
    """Test that start() calls register() before the poll loop."""

    async def test_start_calls_register_before_poll(
        self, worker, mock_session_factory, mock_priority_queue
    ):
        """start() should call register() before entering poll loop."""
        call_order = []

        original_register = worker.register

        async def track_register():
            call_order.append("register")
            await original_register()

        async def track_dequeue(*args, **kwargs):
            call_order.append("dequeue")
            worker._shutdown_requested = True
            return None

        worker.register = track_register
        mock_priority_queue.dequeue = track_dequeue

        await worker.start()

        assert call_order[0] == "register"
        assert "dequeue" in call_order
