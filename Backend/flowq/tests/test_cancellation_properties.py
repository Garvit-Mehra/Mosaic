"""Property-based tests for cancellation state guard and idempotence.

**Validates: Requirements 3.1, 3.2, 3.3**

Property 11: Cancellation State Guard
- For ALL cancellable states (PENDING, SCHEDULED, QUEUED), cancellation returns 200
  and transitions to CANCELLED.
- For ALL non-cancellable states (RUNNING, COMPLETED, FAILED, DEAD_LETTER),
  cancellation returns 409 and does not modify job state.

Property 12: Cancellation Idempotence
- Cancelling an already-CANCELLED job any number of times always returns 200
  without side effects.
"""

import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from src.core.coordinator import JobCoordinator
from src.models.enums import JobStatus
from src.models.job import Job


# --- Strategies ---

CANCELLABLE_STATES = [JobStatus.PENDING, JobStatus.SCHEDULED, JobStatus.QUEUED]
NON_CANCELLABLE_STATES = [JobStatus.RUNNING, JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.DEAD_LETTER]

cancellable_status_strategy = st.sampled_from(CANCELLABLE_STATES)
non_cancellable_status_strategy = st.sampled_from(NON_CANCELLABLE_STATES)
repeat_count_strategy = st.integers(min_value=2, max_value=10)


# --- Helpers ---

def _make_job(status: JobStatus, job_id: uuid.UUID = None) -> MagicMock:
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
    session = AsyncMock()
    session_ctx = AsyncMock()
    session_ctx.__aenter__ = AsyncMock(return_value=session)
    session_ctx.__aexit__ = AsyncMock(return_value=False)

    begin_ctx = AsyncMock()
    begin_ctx.__aenter__ = AsyncMock(return_value=None)
    begin_ctx.__aexit__ = AsyncMock(return_value=False)
    session.begin = MagicMock(return_value=begin_ctx)

    result_mock = MagicMock()
    result_mock.scalar_one_or_none = MagicMock(return_value=job)
    session.execute = AsyncMock(return_value=result_mock)

    session_factory = MagicMock(return_value=session_ctx)

    redis_client = AsyncMock()
    redis_client.zrem = AsyncMock(return_value=1)

    priority_queue = AsyncMock()
    priority_queue.remove = AsyncMock(return_value=True)

    coordinator = JobCoordinator(
        session_factory=session_factory,
        redis_client=redis_client,
        priority_queue=priority_queue,
    )

    return coordinator, redis_client, priority_queue


# --- Property 11: Cancellation State Guard ---


class TestCancellationStateGuardProperty:
    """Property 11: Cancellation State Guard.

    **Validates: Requirements 3.1, 3.2**

    For ALL cancellable states (PENDING, SCHEDULED, QUEUED), cancellation
    returns 200 and transitions to CANCELLED.

    For ALL non-cancellable states (RUNNING, COMPLETED, FAILED, DEAD_LETTER),
    cancellation returns 409 and does not modify job state.
    """

    @pytest.mark.asyncio
    @given(status=cancellable_status_strategy)
    @settings(max_examples=50, deadline=None)
    async def test_cancellable_states_return_200_and_transition(self, status: JobStatus):
        """For any cancellable state, cancel_job returns 200 and sets CANCELLED."""
        job = _make_job(status)
        coordinator, _, _ = _make_coordinator(job=job)

        result = await coordinator.cancel_job(job.id)

        assert result["status_code"] == 200, (
            f"Expected 200 for cancellable state {status.value}, got {result['status_code']}"
        )
        assert result["success"] is True
        assert job.status == JobStatus.CANCELLED, (
            f"Expected job to transition to CANCELLED from {status.value}, "
            f"but got {job.status}"
        )

    @pytest.mark.asyncio
    @given(status=non_cancellable_status_strategy)
    @settings(max_examples=50, deadline=None)
    async def test_non_cancellable_states_return_409_and_preserve_state(self, status: JobStatus):
        """For any non-cancellable state, cancel_job returns 409 without modifying state."""
        job = _make_job(status)
        original_status = job.status
        coordinator, redis_client, priority_queue = _make_coordinator(job=job)

        result = await coordinator.cancel_job(job.id)

        assert result["status_code"] == 409, (
            f"Expected 409 for non-cancellable state {status.value}, got {result['status_code']}"
        )
        assert result["error"] == "not_cancellable"
        assert job.status == original_status, (
            f"Expected state to remain {original_status.value}, "
            f"but it changed to {job.status}"
        )
        # No Redis operations should have been performed
        redis_client.zrem.assert_not_awaited()
        priority_queue.remove.assert_not_awaited()


# --- Property 12: Cancellation Idempotence ---


class TestCancellationIdempotenceProperty:
    """Property 12: Cancellation Idempotence.

    **Validates: Requirements 3.3**

    Cancelling an already-CANCELLED job any number of times always returns 200
    without side effects (no Redis modifications, no state changes).
    """

    @pytest.mark.asyncio
    @given(repeat_count=repeat_count_strategy)
    @settings(max_examples=30, deadline=None)
    async def test_repeated_cancellation_of_cancelled_job_is_idempotent(self, repeat_count: int):
        """Cancelling a CANCELLED job N times always returns 200 with no side effects."""
        job = _make_job(JobStatus.CANCELLED)
        coordinator, redis_client, priority_queue = _make_coordinator(job=job)

        for i in range(repeat_count):
            result = await coordinator.cancel_job(job.id)

            assert result["status_code"] == 200, (
                f"Expected 200 on cancellation attempt {i + 1}, got {result['status_code']}"
            )
            assert result["success"] is True
            assert job.status == JobStatus.CANCELLED, (
                f"Expected status to remain CANCELLED on attempt {i + 1}, "
                f"but got {job.status}"
            )

        # After all repetitions, no Redis operations should have occurred
        redis_client.zrem.assert_not_awaited()
        priority_queue.remove.assert_not_awaited()

    @pytest.mark.asyncio
    @given(status=cancellable_status_strategy, repeat_count=repeat_count_strategy)
    @settings(max_examples=30, deadline=None)
    async def test_cancel_then_repeat_is_idempotent(self, status: JobStatus, repeat_count: int):
        """First cancel transitions state, subsequent cancels return 200 idempotently."""
        job = _make_job(status)
        coordinator, redis_client, priority_queue = _make_coordinator(job=job)

        # First cancellation should succeed and transition to CANCELLED
        first_result = await coordinator.cancel_job(job.id)
        assert first_result["status_code"] == 200
        assert first_result["success"] is True
        assert job.status == JobStatus.CANCELLED

        # Subsequent cancellations: job is now CANCELLED, should return 200 idempotently
        for i in range(repeat_count):
            result = await coordinator.cancel_job(job.id)

            assert result["status_code"] == 200, (
                f"Expected 200 on repeat cancellation {i + 1}, got {result['status_code']}"
            )
            assert result["success"] is True
            assert job.status == JobStatus.CANCELLED
