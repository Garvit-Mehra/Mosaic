"""Tests for worker graceful shutdown behavior.

Tests cover signal handling, status transitions, heartbeat cleanup,
and current job completion during shutdown.

Requirements tested:
- Req 6.6: Handle SIGTERM/SIGINT, stop accepting new jobs, finish current job
- Req 7.4: Update status to SHUTTING_DOWN, finish job, remove heartbeat key
"""

import asyncio
import signal
import sys
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.core.distributed_lock import DistributedLock
from src.core.handler_registry import HandlerRegistry
from src.core.priority_queue import PriorityQueueInterface
from src.models.enums import WorkerStatus
from src.workers.worker import Worker


# --- Helpers ---


def create_session_factory_with_worker_record(worker_record):
    """Create a mock session factory that returns a worker record."""
    session = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none = MagicMock(return_value=worker_record)
    session.execute = AsyncMock(return_value=result)
    session.add = MagicMock()
    session.flush = AsyncMock()

    begin_ctx = AsyncMock()
    begin_ctx.__aenter__ = AsyncMock(return_value=None)
    begin_ctx.__aexit__ = AsyncMock(return_value=False)
    session.begin = MagicMock(return_value=begin_ctx)

    ctx_manager = AsyncMock()
    ctx_manager.__aenter__ = AsyncMock(return_value=session)
    ctx_manager.__aexit__ = AsyncMock(return_value=False)

    factory = MagicMock(return_value=ctx_manager)
    return factory, session


# --- Fixtures ---


@pytest.fixture
def worker_id():
    return uuid4()


@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    redis = AsyncMock()
    redis.set = AsyncMock(return_value=True)
    redis.delete = AsyncMock(return_value=1)
    return redis


@pytest.fixture
def mock_priority_queue():
    """Mock priority queue that returns None (no jobs)."""
    queue = AsyncMock(spec=PriorityQueueInterface)
    queue.dequeue = AsyncMock(return_value=None)
    return queue


@pytest.fixture
def mock_distributed_lock():
    """Mock distributed lock."""
    lock = AsyncMock(spec=DistributedLock)
    lock.acquire_lock = AsyncMock(return_value=True)
    lock.release_lock = AsyncMock(return_value=True)
    return lock


@pytest.fixture
def handler_registry():
    """Real handler registry."""
    return HandlerRegistry()


@pytest.fixture
def mock_worker_record():
    """Create a mock Worker model record."""
    record = MagicMock()
    record.status = WorkerStatus.ACTIVE
    record.current_job_id = None
    return record


@pytest.fixture
def mock_session_factory(mock_worker_record):
    """Create a mock session factory that returns a worker record."""
    factory, session = create_session_factory_with_worker_record(mock_worker_record)
    return factory


@pytest.fixture
def worker(
    worker_id,
    mock_session_factory,
    mock_redis,
    mock_priority_queue,
    mock_distributed_lock,
    handler_registry,
):
    """Create a Worker instance with mocked dependencies."""
    return Worker(
        worker_id=worker_id,
        session_factory=mock_session_factory,
        redis_client=mock_redis,
        priority_queue=mock_priority_queue,
        distributed_lock=mock_distributed_lock,
        handler_registry=handler_registry,
    )


# --- Test Classes ---


class TestGracefulShutdownBasics:
    """Test basic graceful_shutdown() behavior (Req 6.6, 7.4)."""

    async def test_graceful_shutdown_sets_shutdown_requested(self, worker):
        """graceful_shutdown() should set _shutdown_requested = True."""
        assert worker._shutdown_requested is False
        await worker.graceful_shutdown()
        assert worker._shutdown_requested is True

    async def test_graceful_shutdown_stops_accepting_new_jobs(
        self, worker, mock_priority_queue
    ):
        """After graceful_shutdown(), the poll loop should not dequeue (Req 6.6)."""
        await worker.graceful_shutdown()

        # Poll loop should exit immediately since _shutdown_requested is True
        await worker._poll_loop()
        mock_priority_queue.dequeue.assert_not_called()

    async def test_graceful_shutdown_updates_status_to_shutting_down(
        self, worker, mock_worker_record
    ):
        """graceful_shutdown() should update worker status to SHUTTING_DOWN (Req 7.4)."""
        statuses_set = []

        # Track all status assignments
        original_setattr = type(mock_worker_record).__setattr__

        def track_status(self_obj, name, value):
            if name == "status":
                statuses_set.append(value)
            original_setattr(self_obj, name, value)

        with patch.object(type(mock_worker_record), "__setattr__", track_status):
            await worker.graceful_shutdown()

        # SHUTTING_DOWN should have been set before IDLE
        assert WorkerStatus.SHUTTING_DOWN in statuses_set
        assert statuses_set.index(WorkerStatus.SHUTTING_DOWN) < statuses_set.index(WorkerStatus.IDLE)

    async def test_graceful_shutdown_removes_heartbeat_key(
        self, worker, mock_redis, worker_id
    ):
        """graceful_shutdown() should remove heartbeat:{worker_id} from Redis (Req 7.4)."""
        await worker.graceful_shutdown()
        mock_redis.delete.assert_called_once_with(f"heartbeat:{worker_id}")

    async def test_graceful_shutdown_updates_final_status_to_idle(
        self, worker, mock_worker_record
    ):
        """graceful_shutdown() should update final status to IDLE (Req 7.4)."""
        await worker.graceful_shutdown()
        # After the shutdown sequence, the status should be IDLE
        assert mock_worker_record.status == WorkerStatus.IDLE

    async def test_graceful_shutdown_clears_current_job_id_on_record(
        self, worker, mock_worker_record
    ):
        """graceful_shutdown() should clear current_job_id on worker record."""
        mock_worker_record.current_job_id = uuid4()
        await worker.graceful_shutdown()
        assert mock_worker_record.current_job_id is None

    async def test_graceful_shutdown_idempotent(self, worker, mock_redis):
        """Calling graceful_shutdown() twice should not double-execute."""
        await worker.graceful_shutdown()
        # Reset mock to verify second call doesn't repeat actions
        mock_redis.delete.reset_mock()

        await worker.graceful_shutdown()
        # Should not delete heartbeat again (already shut down)
        mock_redis.delete.assert_not_called()


class TestGracefulShutdownWithCurrentJob:
    """Test graceful_shutdown() waits for current job to finish (Req 6.6)."""

    async def test_waits_for_current_job_to_finish(
        self, worker, mock_redis, mock_worker_record
    ):
        """graceful_shutdown() should wait for _current_job_id to be cleared."""
        job_id = uuid4()
        worker._current_job_id = job_id

        # Simulate the job finishing after a delay
        async def clear_job_after_delay():
            await asyncio.sleep(0.2)
            worker._current_job_id = None

        asyncio.create_task(clear_job_after_delay())

        # This should block until the job is done, then proceed
        await worker.graceful_shutdown()

        # Verify it waited and then completed (heartbeat removed)
        mock_redis.delete.assert_called_once()
        assert mock_worker_record.status == WorkerStatus.IDLE

    async def test_does_not_remove_heartbeat_until_job_finishes(
        self, worker, mock_redis
    ):
        """Heartbeat should remain until current job finishes."""
        job_id = uuid4()
        worker._current_job_id = job_id
        heartbeat_removed = False

        original_delete = mock_redis.delete

        async def track_delete(*args, **kwargs):
            nonlocal heartbeat_removed
            # At this point, current job should be done
            assert worker._current_job_id is None
            heartbeat_removed = True
            return await original_delete(*args, **kwargs)

        mock_redis.delete = AsyncMock(side_effect=track_delete)

        # Clear job after short delay
        async def clear_job():
            await asyncio.sleep(0.1)
            worker._current_job_id = None

        asyncio.create_task(clear_job())

        await worker.graceful_shutdown()
        assert heartbeat_removed is True


class TestGracefulShutdownErrorHandling:
    """Test graceful_shutdown() handles errors gracefully."""

    async def test_handles_redis_delete_failure(self, worker, mock_redis):
        """Should not crash if Redis delete fails."""
        mock_redis.delete = AsyncMock(side_effect=Exception("Redis unavailable"))

        # Should not raise
        await worker.graceful_shutdown()
        assert worker._shutdown_requested is True

    async def test_handles_database_update_failure(
        self, worker, mock_redis, worker_id
    ):
        """Should not crash if PostgreSQL update fails."""
        # Make the session factory raise on first call (SHUTTING_DOWN update)
        failing_ctx = AsyncMock()
        failing_ctx.__aenter__ = AsyncMock(side_effect=Exception("DB error"))
        failing_ctx.__aexit__ = AsyncMock(return_value=False)
        worker._session_factory = MagicMock(return_value=failing_ctx)

        # Should not raise
        await worker.graceful_shutdown()
        assert worker._shutdown_requested is True
        # Heartbeat removal should still be attempted
        mock_redis.delete.assert_called_once_with(f"heartbeat:{worker_id}")


class TestSignalHandlerRegistration:
    """Test signal handler registration in start() (Req 6.6)."""

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals only")
    async def test_signal_handlers_are_registered(
        self, worker, mock_priority_queue, mock_redis
    ):
        """start() should register SIGTERM and SIGINT handlers."""
        # We'll patch the register method to skip DB work
        worker.register = AsyncMock()

        # Immediately shut down so start() returns quickly
        async def dequeue_then_stop(timeout=5.0):
            worker._shutdown_requested = True
            return None

        mock_priority_queue.dequeue = AsyncMock(side_effect=dequeue_then_stop)

        with patch("asyncio.get_event_loop") as mock_loop_fn:
            mock_loop = MagicMock()
            mock_loop_fn.return_value = mock_loop

            await worker.start()

            # Verify signal handlers were registered
            assert mock_loop.add_signal_handler.call_count == 2
            signals_registered = [
                call[0][0] for call in mock_loop.add_signal_handler.call_args_list
            ]
            assert signal.SIGTERM in signals_registered
            assert signal.SIGINT in signals_registered

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals only")
    async def test_register_signal_handlers_method(self, worker):
        """_register_signal_handlers() should call add_signal_handler for SIGTERM and SIGINT."""
        with patch("asyncio.get_event_loop") as mock_loop_fn:
            mock_loop = MagicMock()
            mock_loop_fn.return_value = mock_loop

            worker._register_signal_handlers()

            assert mock_loop.add_signal_handler.call_count == 2
            signals_registered = [
                call[0][0] for call in mock_loop.add_signal_handler.call_args_list
            ]
            assert signal.SIGTERM in signals_registered
            assert signal.SIGINT in signals_registered


class TestGracefulShutdownIntegration:
    """Integration-style tests for the complete shutdown flow."""

    async def test_shutdown_during_poll_loop(
        self, worker, mock_priority_queue, mock_redis, mock_worker_record
    ):
        """Triggering graceful_shutdown during poll loop stops accepting jobs."""
        call_count = 0

        async def dequeue_with_shutdown(timeout=5.0):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                # Trigger shutdown on second dequeue attempt
                asyncio.ensure_future(worker.graceful_shutdown())
                await asyncio.sleep(0.05)
            return None

        mock_priority_queue.dequeue = AsyncMock(side_effect=dequeue_with_shutdown)

        await worker._poll_loop()

        # Worker should have stopped accepting new jobs
        assert worker._shutdown_requested is True
        # Heartbeat should be removed
        mock_redis.delete.assert_called_once()

    async def test_shutdown_finishes_current_job_then_exits(
        self, worker, mock_priority_queue, mock_distributed_lock,
        mock_redis, handler_registry, mock_worker_record
    ):
        """Worker finishes executing current job before shutdown completes."""
        job_id = uuid4()
        job_finished = False

        # A handler that takes some time
        async def slow_handler(payload):
            nonlocal job_finished
            await asyncio.sleep(0.3)
            job_finished = True
            return {"done": True}

        handler_registry.register("test_job", slow_handler)

        # Mock job
        mock_job = MagicMock()
        mock_job.id = job_id
        mock_job.job_type = "test_job"
        mock_job.payload = {"key": "value"}
        mock_job.timeout_seconds = 60
        mock_job.retry_count = 0
        mock_job.status = "QUEUED"

        factory, session = create_session_factory_with_worker_record(mock_job)
        worker._session_factory = factory

        # Trigger graceful shutdown after job starts
        async def trigger_shutdown():
            await asyncio.sleep(0.1)  # Let job start executing
            # Directly set _current_job_id to None later since our mock
            # won't clear it - the actual execute_job clears it in finally
            await worker.graceful_shutdown()

        # Simulate the worker executing a job while shutdown is requested
        worker._current_job_id = job_id
        shutdown_task = asyncio.create_task(trigger_shutdown())

        # Simulate job completion after a delay
        async def complete_job():
            await asyncio.sleep(0.2)
            worker._current_job_id = None

        asyncio.create_task(complete_job())

        await shutdown_task

        # Shutdown should have waited for the job
        assert worker._shutdown_requested is True
        mock_redis.delete.assert_called_once()
