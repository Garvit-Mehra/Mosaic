"""Unit tests for job cancellation logic (Requirements 3.1-3.4)."""

import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.coordinator import JobCoordinator, SCHEDULED_QUEUE_KEY
from src.models.enums import JobStatus
from src.models.job import Job


def _make_job(status: JobStatus, job_id: uuid.UUID = None) -> Job:
    """Create a mock Job instance with the given status."""
    job = MagicMock(spec=Job)
    job.id = job_id or uuid.uuid4()
    job.job_type = "test_job"
    job.payload = {"key": "value"}
    job.priority = 5
    job.status = status
    job.execute_at = None
    job.max_retries = 3
    job.retry_count = 0
    job.retry_backoff_base = 2.0
    job.timeout_seconds = 300
    job.worker_id = None
    job.started_at = None
    job.completed_at = None
    job.result = None
    job.error = None
    job.created_at = datetime.utcnow()
    job.updated_at = datetime.utcnow()
    return job


def _make_coordinator(job=None):
    """Create a JobCoordinator with mocked dependencies.

    If `job` is provided, the session will return it on query.
    If `job` is None, the session will return None (not found).
    """
    # Mock session and session factory
    session = AsyncMock()
    session_ctx = AsyncMock()
    session_ctx.__aenter__ = AsyncMock(return_value=session)
    session_ctx.__aexit__ = AsyncMock(return_value=False)

    # Mock the begin() context manager
    begin_ctx = AsyncMock()
    begin_ctx.__aenter__ = AsyncMock(return_value=None)
    begin_ctx.__aexit__ = AsyncMock(return_value=False)
    session.begin = MagicMock(return_value=begin_ctx)

    # Mock execute result
    result_mock = MagicMock()
    result_mock.scalar_one_or_none = MagicMock(return_value=job)
    session.execute = AsyncMock(return_value=result_mock)

    session_factory = MagicMock(return_value=session_ctx)

    # Mock Redis client
    redis_client = AsyncMock()
    redis_client.zrem = AsyncMock(return_value=1)

    # Mock priority queue
    priority_queue = AsyncMock()
    priority_queue.remove = AsyncMock(return_value=True)

    coordinator = JobCoordinator(
        session_factory=session_factory,
        redis_client=redis_client,
        priority_queue=priority_queue,
    )

    return coordinator, redis_client, priority_queue


class TestCancelJobNotFound:
    """Req 3.4: Non-existent job returns 404."""

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_job_returns_404(self):
        """Cancelling a job that doesn't exist returns 404."""
        coordinator, _, _ = _make_coordinator(job=None)
        job_id = uuid.uuid4()

        result = await coordinator.cancel_job(job_id)

        assert result["status_code"] == 404
        assert result["error"] == "not_found"


class TestCancelJobIdempotent:
    """Req 3.3: Already CANCELLED job returns 200 without modification."""

    @pytest.mark.asyncio
    async def test_cancel_already_cancelled_returns_200(self):
        """Cancelling an already-cancelled job returns 200 idempotently."""
        job = _make_job(JobStatus.CANCELLED)
        coordinator, redis_client, priority_queue = _make_coordinator(job=job)

        result = await coordinator.cancel_job(job.id)

        assert result["status_code"] == 200
        assert result["success"] is True
        # Should not touch Redis
        redis_client.zrem.assert_not_awaited()
        priority_queue.remove.assert_not_awaited()


class TestCancelJobNotCancellable:
    """Req 3.2: RUNNING, COMPLETED, FAILED, DEAD_LETTER return 409."""

    @pytest.mark.asyncio
    async def test_cancel_running_job_returns_409(self):
        """Cancelling a RUNNING job returns 409."""
        job = _make_job(JobStatus.RUNNING)
        coordinator, _, _ = _make_coordinator(job=job)

        result = await coordinator.cancel_job(job.id)

        assert result["status_code"] == 409
        assert result["error"] == "not_cancellable"

    @pytest.mark.asyncio
    async def test_cancel_completed_job_returns_409(self):
        """Cancelling a COMPLETED job returns 409."""
        job = _make_job(JobStatus.COMPLETED)
        coordinator, _, _ = _make_coordinator(job=job)

        result = await coordinator.cancel_job(job.id)

        assert result["status_code"] == 409
        assert result["error"] == "not_cancellable"

    @pytest.mark.asyncio
    async def test_cancel_failed_job_returns_409(self):
        """Cancelling a FAILED job returns 409."""
        job = _make_job(JobStatus.FAILED)
        coordinator, _, _ = _make_coordinator(job=job)

        result = await coordinator.cancel_job(job.id)

        assert result["status_code"] == 409
        assert result["error"] == "not_cancellable"

    @pytest.mark.asyncio
    async def test_cancel_dead_letter_job_returns_409(self):
        """Cancelling a DEAD_LETTER job returns 409."""
        job = _make_job(JobStatus.DEAD_LETTER)
        coordinator, _, _ = _make_coordinator(job=job)

        result = await coordinator.cancel_job(job.id)

        assert result["status_code"] == 409
        assert result["error"] == "not_cancellable"


class TestCancelJobCancellableStates:
    """Req 3.1: PENDING, SCHEDULED, QUEUED can be cancelled."""

    @pytest.mark.asyncio
    async def test_cancel_pending_job_returns_200(self):
        """Cancelling a PENDING job transitions to CANCELLED and returns 200."""
        job = _make_job(JobStatus.PENDING)
        coordinator, redis_client, priority_queue = _make_coordinator(job=job)

        result = await coordinator.cancel_job(job.id)

        assert result["status_code"] == 200
        assert result["success"] is True
        # PENDING: no Redis removal needed
        redis_client.zrem.assert_not_awaited()
        priority_queue.remove.assert_not_awaited()
        # State machine transition applied
        assert job.status == JobStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_queued_job_removes_from_priority_queue(self):
        """Cancelling a QUEUED job removes it from the priority queue."""
        job = _make_job(JobStatus.QUEUED)
        coordinator, redis_client, priority_queue = _make_coordinator(job=job)

        result = await coordinator.cancel_job(job.id)

        assert result["status_code"] == 200
        assert result["success"] is True
        # Should remove from priority queue
        priority_queue.remove.assert_awaited_once_with(job.id)
        # Should NOT touch scheduled queue
        redis_client.zrem.assert_not_awaited()
        # State machine transition applied
        assert job.status == JobStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_scheduled_job_removes_from_schedule_set(self):
        """Cancelling a SCHEDULED job removes it from the Redis schedule set."""
        job = _make_job(JobStatus.SCHEDULED)
        coordinator, redis_client, priority_queue = _make_coordinator(job=job)

        result = await coordinator.cancel_job(job.id)

        assert result["status_code"] == 200
        assert result["success"] is True
        # Should remove from scheduled queue
        redis_client.zrem.assert_awaited_once_with(SCHEDULED_QUEUE_KEY, str(job.id))
        # Should NOT touch priority queue
        priority_queue.remove.assert_not_awaited()
        # State machine transition applied
        assert job.status == JobStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_job_updates_timestamp(self):
        """Cancelling a job updates the updated_at timestamp."""
        job = _make_job(JobStatus.PENDING)
        original_updated_at = job.updated_at
        coordinator, _, _ = _make_coordinator(job=job)

        result = await coordinator.cancel_job(job.id)

        assert result["status_code"] == 200
        # The state machine apply_transition sets updated_at
        assert job.updated_at != original_updated_at
