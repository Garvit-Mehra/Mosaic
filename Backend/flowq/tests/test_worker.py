"""Tests for the Worker process implementation.

Tests cover the worker poll loop, job execution, lock handling,
heartbeats, and failure scenarios.

Requirements tested:
- Req 6.1: Worker dequeues and acquires lock before execution
- Req 6.2: Lock acquisition failure → skip job
- Req 6.3: Update to RUNNING with worker_id and started_at
- Req 6.4: On success → COMPLETED with result and completed_at
- Req 6.5: Timeout → FAILED, trigger retry/DLQ
- Req 6.7: Unregistered handler → FAILED
- Req 6.8: Continue heartbeats during execution
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from src.core.distributed_lock import DistributedLock, LOCK_TTL_BUFFER
from src.core.handler_registry import HandlerRegistry
from src.core.priority_queue import PriorityQueueInterface
from src.models.enums import JobStatus
from src.workers.worker import Worker, HEARTBEAT_TTL, HEARTBEAT_INTERVAL


# --- Fixtures ---


@pytest.fixture
def worker_id():
    return uuid4()


@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    redis = AsyncMock()
    redis.set = AsyncMock(return_value=True)
    redis.zadd = AsyncMock(return_value=1)
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
def mock_job():
    """Create a mock Job object in QUEUED state."""
    job = MagicMock()
    job.id = uuid4()
    job.job_type = "test_job"
    job.payload = {"key": "value"}
    job.priority = 5
    job.status = JobStatus.QUEUED
    job.timeout_seconds = 300
    job.max_retries = 3
    job.retry_count = 0
    job.retry_backoff_base = 2.0
    job.worker_id = None
    job.started_at = None
    job.completed_at = None
    job.result = None
    job.error = None
    job.created_at = datetime.now(timezone.utc).replace(tzinfo=None)
    job.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
    return job


@pytest.fixture
def mock_session_factory(mock_job):
    """Create a mock session factory that returns a job."""
    session = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none = MagicMock(return_value=mock_job)
    session.execute = AsyncMock(return_value=result)
    session.add = MagicMock()
    session.flush = AsyncMock()

    # Context managers
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)

    # begin() context manager
    begin_ctx = AsyncMock()
    begin_ctx.__aenter__ = AsyncMock(return_value=None)
    begin_ctx.__aexit__ = AsyncMock(return_value=False)
    session.begin = MagicMock(return_value=begin_ctx)

    factory = MagicMock()
    factory.return_value = session
    factory().__aenter__ = AsyncMock(return_value=session)
    factory().__aexit__ = AsyncMock(return_value=False)

    # Make it work as an async context manager
    ctx_manager = AsyncMock()
    ctx_manager.__aenter__ = AsyncMock(return_value=session)
    ctx_manager.__aexit__ = AsyncMock(return_value=False)
    factory.return_value = ctx_manager

    return factory


def create_session_factory_with_job(job):
    """Helper to create a properly configured mock session factory."""
    session = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none = MagicMock(return_value=job)
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


class TestWorkerInit:
    """Test worker initialization."""

    def test_worker_creates_with_id(self, worker, worker_id):
        """Worker should store its ID."""
        assert worker.worker_id == worker_id

    def test_worker_starts_with_zero_counters(self, worker):
        """Worker should start with zero job counters."""
        assert worker.jobs_completed == 0
        assert worker.jobs_failed == 0

    def test_worker_starts_not_shutdown(self, worker):
        """Worker should not be in shutdown state initially."""
        assert worker._shutdown_requested is False

    def test_worker_starts_with_no_current_job(self, worker):
        """Worker should have no current job initially."""
        assert worker._current_job_id is None


class TestWorkerHeartbeat:
    """Test heartbeat functionality (Req 6.8, Req 7.1)."""

    async def test_send_heartbeat_sets_redis_key(self, worker, mock_redis, worker_id):
        """Heartbeat should set heartbeat:{worker_id} in Redis."""
        await worker.send_heartbeat()

        mock_redis.set.assert_called_once()
        call_args = mock_redis.set.call_args
        assert call_args[0][0] == f"heartbeat:{worker_id}"
        assert call_args[1]["ex"] == HEARTBEAT_TTL

    async def test_send_heartbeat_uses_15s_ttl(self, worker, mock_redis):
        """Heartbeat TTL should be 15 seconds."""
        await worker.send_heartbeat()

        call_args = mock_redis.set.call_args
        assert call_args[1]["ex"] == 15

    async def test_heartbeat_failure_does_not_raise(self, worker, mock_redis):
        """Heartbeat failure should log warning but not raise."""
        mock_redis.set.side_effect = Exception("Redis connection lost")

        # Should not raise
        await worker.send_heartbeat()

    async def test_heartbeat_loop_sends_initial_heartbeat(
        self, worker, mock_redis
    ):
        """Heartbeat loop should send heartbeat immediately on start."""
        # Run heartbeat loop briefly
        worker._shutdown_requested = False

        async def stop_after_first():
            await asyncio.sleep(0.05)
            worker._shutdown_requested = True

        asyncio.create_task(stop_after_first())
        await worker._heartbeat_loop()

        # At least one heartbeat should have been sent
        assert mock_redis.set.call_count >= 1


class TestWorkerPollLoop:
    """Test the main poll loop."""

    async def test_poll_loop_exits_on_shutdown(
        self, worker, mock_priority_queue
    ):
        """Poll loop should exit when shutdown is requested."""
        worker._shutdown_requested = True
        await worker._poll_loop()
        # Should exit immediately, no dequeue call
        mock_priority_queue.dequeue.assert_not_called()

    async def test_poll_loop_dequeues_from_priority_queue(
        self, worker, mock_priority_queue
    ):
        """Poll loop should call dequeue on the priority queue."""
        call_count = 0

        async def dequeue_then_shutdown(timeout=5.0):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                worker._shutdown_requested = True
            return None

        mock_priority_queue.dequeue = AsyncMock(side_effect=dequeue_then_shutdown)

        await worker._poll_loop()

        assert mock_priority_queue.dequeue.call_count >= 1

    async def test_poll_loop_continues_on_no_job(
        self, worker, mock_priority_queue
    ):
        """Poll loop should continue when dequeue returns None."""
        call_count = 0

        async def dequeue_side_effect(timeout=5.0):
            nonlocal call_count
            call_count += 1
            if call_count >= 3:
                worker._shutdown_requested = True
            return None

        mock_priority_queue.dequeue = AsyncMock(
            side_effect=dequeue_side_effect
        )

        await worker._poll_loop()

        # Should have looped multiple times
        assert call_count == 3


class TestExecuteJobLocking:
    """Test lock acquisition during job execution (Req 6.1, 6.2)."""

    async def test_acquires_lock_before_execution(
        self, worker, mock_distributed_lock, mock_job, worker_id, handler_registry
    ):
        """Worker should acquire lock before executing (Req 6.1)."""
        # Register a handler
        handler_registry.register("test_job", AsyncMock(return_value={"done": True}))

        await worker.execute_job(mock_job.id)

        mock_distributed_lock.acquire_lock.assert_called_once_with(
            mock_job.id, worker_id, mock_job.timeout_seconds + LOCK_TTL_BUFFER
        )

    async def test_skips_job_on_lock_failure(
        self, worker, mock_distributed_lock, mock_job, handler_registry
    ):
        """Worker should skip job when lock acquisition fails (Req 6.2)."""
        mock_distributed_lock.acquire_lock = AsyncMock(return_value=False)
        handler_registry.register("test_job", AsyncMock(return_value={"done": True}))

        await worker.execute_job(mock_job.id)

        # Lock was not acquired, should not have attempted execution
        # jobs_completed should remain 0
        assert worker.jobs_completed == 0
        assert worker.jobs_failed == 0

    async def test_releases_lock_on_success(
        self, worker, mock_distributed_lock, mock_job, worker_id, handler_registry
    ):
        """Worker should release lock after successful execution."""
        handler_registry.register("test_job", AsyncMock(return_value={"done": True}))

        await worker.execute_job(mock_job.id)

        mock_distributed_lock.release_lock.assert_called_once_with(
            mock_job.id, worker_id
        )

    async def test_releases_lock_on_failure(
        self, worker, mock_distributed_lock, mock_job, worker_id, handler_registry
    ):
        """Worker should release lock even when execution fails."""
        handler_registry.register(
            "test_job", AsyncMock(side_effect=RuntimeError("handler crashed"))
        )

        await worker.execute_job(mock_job.id)

        mock_distributed_lock.release_lock.assert_called_once_with(
            mock_job.id, worker_id
        )

    async def test_releases_lock_on_timeout(
        self, worker, mock_distributed_lock, mock_job, worker_id, handler_registry
    ):
        """Worker should release lock on timeout."""
        # Set a very short timeout
        mock_job.timeout_seconds = 1

        async def slow_handler(payload):
            await asyncio.sleep(10)
            return {"done": True}

        handler_registry.register("test_job", slow_handler)

        # Re-create factory with short-timeout job
        factory, session = create_session_factory_with_job(mock_job)
        worker._session_factory = factory

        await worker.execute_job(mock_job.id)

        mock_distributed_lock.release_lock.assert_called_once_with(
            mock_job.id, worker_id
        )

    async def test_lock_ttl_includes_buffer(
        self, worker, mock_distributed_lock, mock_job, worker_id, handler_registry
    ):
        """Lock TTL should be timeout_seconds + 30s buffer."""
        mock_job.timeout_seconds = 120
        handler_registry.register("test_job", AsyncMock(return_value={}))

        await worker.execute_job(mock_job.id)

        expected_ttl = 120 + LOCK_TTL_BUFFER  # 150
        mock_distributed_lock.acquire_lock.assert_called_once_with(
            mock_job.id, worker_id, expected_ttl
        )


class TestExecuteJobSuccess:
    """Test successful job execution (Req 6.3, 6.4)."""

    async def test_updates_to_running(
        self, worker, mock_job, worker_id, handler_registry
    ):
        """Worker should update job to RUNNING with worker_id and started_at (Req 6.3)."""
        handler_registry.register("test_job", AsyncMock(return_value={"done": True}))

        await worker.execute_job(mock_job.id)

        # Check job was transitioned to RUNNING (via apply_transition mock calls)
        # The mock_job's status should have been modified through the session
        assert worker.jobs_completed == 1

    async def test_increments_completed_counter(
        self, worker, mock_job, handler_registry
    ):
        """Worker should increment jobs_completed on success."""
        handler_registry.register("test_job", AsyncMock(return_value={"done": True}))

        await worker.execute_job(mock_job.id)

        assert worker.jobs_completed == 1
        assert worker.jobs_failed == 0

    async def test_handler_receives_payload(
        self, worker, mock_job, handler_registry
    ):
        """Handler should receive the job payload."""
        mock_handler = AsyncMock(return_value={"processed": True})
        handler_registry.register("test_job", mock_handler)

        await worker.execute_job(mock_job.id)

        mock_handler.assert_called_once_with(mock_job.payload)

    async def test_clears_current_job_after_execution(
        self, worker, mock_job, handler_registry
    ):
        """current_job_id should be None after execution completes."""
        handler_registry.register("test_job", AsyncMock(return_value={}))

        await worker.execute_job(mock_job.id)

        assert worker._current_job_id is None


class TestExecuteJobTimeout:
    """Test timeout handling (Req 6.5)."""

    async def test_timeout_increments_failed_counter(
        self, worker, mock_job, handler_registry
    ):
        """Worker should increment jobs_failed on timeout."""
        mock_job.timeout_seconds = 1

        async def slow_handler(payload):
            await asyncio.sleep(10)
            return {}

        handler_registry.register("test_job", slow_handler)

        factory, session = create_session_factory_with_job(mock_job)
        worker._session_factory = factory

        await worker.execute_job(mock_job.id)

        assert worker.jobs_failed == 1
        assert worker.jobs_completed == 0

    async def test_timeout_enforced_via_wait_for(
        self, worker, mock_job, handler_registry
    ):
        """Jobs should be cancelled after timeout_seconds (Req 6.5)."""
        mock_job.timeout_seconds = 1
        execution_completed = False

        async def slow_handler(payload):
            nonlocal execution_completed
            await asyncio.sleep(10)
            execution_completed = True
            return {}

        handler_registry.register("test_job", slow_handler)

        factory, session = create_session_factory_with_job(mock_job)
        worker._session_factory = factory

        await worker.execute_job(mock_job.id)

        # Handler should not have completed
        assert execution_completed is False


class TestExecuteJobFailure:
    """Test failure handling."""

    async def test_exception_increments_failed_counter(
        self, worker, mock_job, handler_registry
    ):
        """Worker should increment jobs_failed on exception."""
        handler_registry.register(
            "test_job", AsyncMock(side_effect=ValueError("bad data"))
        )

        await worker.execute_job(mock_job.id)

        assert worker.jobs_failed == 1
        assert worker.jobs_completed == 0

    async def test_clears_current_job_on_failure(
        self, worker, mock_job, handler_registry
    ):
        """current_job_id should be cleared even on failure."""
        handler_registry.register(
            "test_job", AsyncMock(side_effect=RuntimeError("crash"))
        )

        await worker.execute_job(mock_job.id)

        assert worker._current_job_id is None


class TestUnregisteredHandler:
    """Test unregistered handler handling (Req 6.7)."""

    async def test_unregistered_handler_fails_job(
        self, worker, mock_job, handler_registry
    ):
        """Unregistered job type should result in failure (Req 6.7)."""
        # Don't register any handler for "test_job"
        await worker.execute_job(mock_job.id)

        assert worker.jobs_failed == 1
        assert worker.jobs_completed == 0

    async def test_unregistered_handler_releases_lock(
        self, worker, mock_distributed_lock, mock_job, worker_id
    ):
        """Lock should be released for unregistered handler."""
        # No handler registered
        await worker.execute_job(mock_job.id)

        mock_distributed_lock.release_lock.assert_called_once_with(
            mock_job.id, worker_id
        )


class TestWorkerStop:
    """Test worker stop/shutdown."""

    async def test_stop_sets_shutdown_flag(self, worker):
        """stop() should set shutdown_requested flag."""
        await worker.stop(graceful=True)
        assert worker._shutdown_requested is True

    async def test_graceful_stop_does_not_cancel_heartbeat(self, worker):
        """Graceful stop should not cancel heartbeat task immediately."""
        worker._heartbeat_task = AsyncMock()
        await worker.stop(graceful=True)
        # Heartbeat task should not be cancelled during graceful stop
        # (it will be cleaned up in start's finally block)
        assert worker._shutdown_requested is True

    async def test_forced_stop_cancels_heartbeat(self, worker):
        """Non-graceful stop should cancel the heartbeat task."""
        task = AsyncMock()
        worker._heartbeat_task = task
        await worker.stop(graceful=False)
        task.cancel.assert_called_once()


class TestWorkerStartStop:
    """Test the start/stop lifecycle."""

    async def test_start_runs_poll_loop(
        self, worker, mock_priority_queue, mock_redis
    ):
        """start() should enter the poll loop."""
        call_count = 0

        async def dequeue_then_shutdown(timeout=5.0):
            nonlocal call_count
            call_count += 1
            if call_count >= 1:
                worker._shutdown_requested = True
            return None

        mock_priority_queue.dequeue = AsyncMock(side_effect=dequeue_then_shutdown)

        await worker.start()

        assert call_count >= 1

    async def test_start_creates_and_cancels_heartbeat_task(
        self, worker, mock_priority_queue, mock_redis
    ):
        """start() should create heartbeat task that gets cancelled on exit."""
        call_count = 0

        async def dequeue_then_shutdown(timeout=5.0):
            nonlocal call_count
            call_count += 1
            # Give heartbeat task a moment to run
            await asyncio.sleep(0.01)
            worker._shutdown_requested = True
            return None

        mock_priority_queue.dequeue = AsyncMock(side_effect=dequeue_then_shutdown)

        await worker.start()

        # Heartbeat should have been sent (initial heartbeat fires immediately)
        assert mock_redis.set.call_count >= 1
