"""Property-based tests for retry bound and DLQ completeness.

**Validates: Requirements 9.5, 10.1, 8.3, 8.4**

Uses Hypothesis to verify:
- Property 4 (Retry Bound): retry_count never exceeds max_retries + 1
- Property 10 (DLQ Completeness): Jobs that exhaust retries always reach DEAD_LETTER
- Additional: Jobs with retries remaining are always re-queued (not DLQ'd)
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from src.core.failure_handler import (
    DLQ_KEY,
    RETRY_QUEUE_KEY,
    handle_job_failure,
)
from src.models.enums import JobStatus


@dataclass
class FakeJob:
    """Lightweight job stand-in for testing without DB dependencies."""

    id: uuid.UUID = field(default_factory=uuid.uuid4)
    job_type: str = "test_job"
    status: JobStatus = JobStatus.RUNNING
    retry_count: int = 0
    max_retries: int = 3
    retry_backoff_base: float = 2.0
    worker_id: uuid.UUID = field(default_factory=uuid.uuid4)
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    updated_at: datetime = field(default_factory=datetime.utcnow)


def make_mock_session():
    """Create an AsyncMock session with synchronous add."""
    session = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()
    return session


def make_mock_redis():
    """Create an AsyncMock redis client."""
    redis_client = AsyncMock()
    redis_client.zadd = AsyncMock()
    return redis_client


# --- Strategies ---

# max_retries: 0 to 100 (per requirement 1.5 validation range)
max_retries_strategy = st.integers(min_value=0, max_value=100)

# retry_count: must be valid relative to max_retries (will be constrained per test)
retry_count_strategy = st.integers(min_value=0, max_value=100)

# backoff base: reasonable values that don't overflow
backoff_base_strategy = st.floats(min_value=1.0, max_value=5.0, allow_nan=False, allow_infinity=False)

# number of sequential failures to simulate
failure_sequence_length = st.integers(min_value=1, max_value=20)


# --- Property Tests ---


class TestRetryBound:
    """Property 4: Retry Bound.

    For ANY max_retries value and ANY sequence of failures, retry_count
    never exceeds max_retries + 1.

    **Validates: Requirements 9.5**
    """

    @given(
        max_retries=max_retries_strategy,
        backoff_base=backoff_base_strategy,
    )
    @settings(max_examples=200)
    @pytest.mark.asyncio
    async def test_retry_count_never_exceeds_max_retries_plus_one(
        self, max_retries, backoff_base
    ):
        """After ANY number of failures, retry_count <= max_retries + 1.

        The +1 accounts for the final failure that triggers DLQ routing.
        Once a job reaches DEAD_LETTER, no further failures should occur.

        **Validates: Requirements 9.5**
        """
        session = make_mock_session()
        redis_client = make_mock_redis()

        job = FakeJob(
            retry_count=0,
            max_retries=max_retries,
            retry_backoff_base=backoff_base,
        )

        # Simulate up to max_retries + 2 failures (more than enough to exhaust)
        num_failures = max_retries + 2

        for i in range(num_failures):
            # Only process failures while job is not in a terminal state
            if job.status == JobStatus.DEAD_LETTER:
                break

            # Reset job to RUNNING state to simulate worker picking it up
            job.status = JobStatus.RUNNING
            job.worker_id = uuid.uuid4()
            job.started_at = datetime.utcnow()
            job.updated_at = datetime.utcnow()

            await handle_job_failure(job, f"error #{i+1}", session, redis_client)

            # PROPERTY: retry_count must never exceed max_retries + 1
            assert job.retry_count <= max_retries + 1, (
                f"retry_count {job.retry_count} exceeded max_retries + 1 "
                f"({max_retries + 1}) after {i+1} failures"
            )

    @given(
        max_retries=st.integers(min_value=1, max_value=50),
        initial_retry_count=st.data(),
        backoff_base=backoff_base_strategy,
    )
    @settings(max_examples=200)
    @pytest.mark.asyncio
    async def test_single_failure_never_pushes_retry_count_above_bound(
        self, max_retries, initial_retry_count, backoff_base
    ):
        """A single failure from ANY valid starting retry_count never exceeds the bound.

        **Validates: Requirements 9.5**
        """
        # Draw initial_retry_count constrained to [0, max_retries]
        rc = initial_retry_count.draw(
            st.integers(min_value=0, max_value=max_retries)
        )

        session = make_mock_session()
        redis_client = make_mock_redis()

        job = FakeJob(
            retry_count=rc,
            max_retries=max_retries,
            retry_backoff_base=backoff_base,
        )

        await handle_job_failure(job, "test error", session, redis_client)

        # PROPERTY: After one failure, retry_count = rc + 1, which is at most max_retries + 1
        assert job.retry_count == rc + 1
        assert job.retry_count <= max_retries + 1


class TestDLQCompleteness:
    """Property 10: DLQ Completeness.

    When retry_count >= max_retries and a failure occurs, the job ALWAYS
    ends up in DEAD_LETTER state and is added to queue:dlq.

    **Validates: Requirements 10.1, 8.4**
    """

    @given(
        max_retries=max_retries_strategy,
        extra_retries=st.integers(min_value=0, max_value=10),
        backoff_base=backoff_base_strategy,
    )
    @settings(max_examples=200)
    @pytest.mark.asyncio
    async def test_exhausted_retries_always_reach_dead_letter(
        self, max_retries, extra_retries, backoff_base
    ):
        """Jobs with retry_count >= max_retries ALWAYS transition to DEAD_LETTER.

        **Validates: Requirements 10.1**
        """
        # retry_count at or above max_retries means retries are exhausted
        retry_count = max_retries + extra_retries

        session = make_mock_session()
        redis_client = make_mock_redis()

        job = FakeJob(
            retry_count=retry_count,
            max_retries=max_retries,
            retry_backoff_base=backoff_base,
        )

        await handle_job_failure(job, "exhausted error", session, redis_client)

        # PROPERTY: Job must be in DEAD_LETTER state
        assert job.status == JobStatus.DEAD_LETTER, (
            f"Expected DEAD_LETTER but got {job.status} "
            f"(retry_count={retry_count}, max_retries={max_retries})"
        )

    @given(
        max_retries=max_retries_strategy,
        extra_retries=st.integers(min_value=0, max_value=10),
        backoff_base=backoff_base_strategy,
    )
    @settings(max_examples=200)
    @pytest.mark.asyncio
    async def test_exhausted_retries_always_added_to_dlq_queue(
        self, max_retries, extra_retries, backoff_base
    ):
        """Jobs with exhausted retries are ALWAYS added to queue:dlq in Redis.

        **Validates: Requirements 10.1, 8.4**
        """
        retry_count = max_retries + extra_retries

        session = make_mock_session()
        redis_client = make_mock_redis()

        job = FakeJob(
            retry_count=retry_count,
            max_retries=max_retries,
            retry_backoff_base=backoff_base,
        )

        await handle_job_failure(job, "exhausted error", session, redis_client)

        # PROPERTY: Redis zadd must be called with DLQ key
        redis_client.zadd.assert_called_once()
        call_args = redis_client.zadd.call_args
        assert call_args[0][0] == DLQ_KEY, (
            f"Expected zadd to DLQ key '{DLQ_KEY}' but got '{call_args[0][0]}'"
        )
        # Job ID must be in the score mapping
        score_mapping = call_args[0][1]
        assert str(job.id) in score_mapping

    @given(
        max_retries=max_retries_strategy,
        backoff_base=backoff_base_strategy,
    )
    @settings(max_examples=200)
    @pytest.mark.asyncio
    async def test_full_failure_sequence_ends_in_dead_letter(
        self, max_retries, backoff_base
    ):
        """Simulating max_retries + 1 consecutive failures always ends in DEAD_LETTER.

        **Validates: Requirements 9.5, 10.1**
        """
        session = make_mock_session()
        redis_client = make_mock_redis()

        job = FakeJob(
            retry_count=0,
            max_retries=max_retries,
            retry_backoff_base=backoff_base,
        )

        # Simulate the full failure sequence
        for i in range(max_retries + 1):
            if job.status == JobStatus.DEAD_LETTER:
                break

            # Reset to RUNNING for next failure
            job.status = JobStatus.RUNNING
            job.worker_id = uuid.uuid4()
            job.started_at = datetime.utcnow()
            job.updated_at = datetime.utcnow()

            await handle_job_failure(job, f"failure #{i+1}", session, redis_client)

        # PROPERTY: After max_retries + 1 failures, job MUST be in DEAD_LETTER
        assert job.status == JobStatus.DEAD_LETTER, (
            f"Expected DEAD_LETTER after {max_retries + 1} failures but got {job.status}"
        )


class TestRetryRequeue:
    """Additional Property: Jobs with retries remaining are always re-queued.

    When retry_count < max_retries and a failure occurs, the job ALWAYS
    ends up re-queued (not in DLQ).

    **Validates: Requirements 8.3**
    """

    @given(
        max_retries=st.integers(min_value=1, max_value=100),
        retry_count=st.data(),
        backoff_base=backoff_base_strategy,
    )
    @settings(max_examples=200)
    @pytest.mark.asyncio
    async def test_retries_remaining_always_requeues(
        self, max_retries, retry_count, backoff_base
    ):
        """Jobs with retry_count < max_retries are ALWAYS re-queued to retry queue.

        **Validates: Requirements 8.3**
        """
        # Draw a retry_count that leaves at least one retry remaining
        # new_retry_count = retry_count + 1 must be <= max_retries for retry path
        rc = retry_count.draw(st.integers(min_value=0, max_value=max_retries - 1))

        session = make_mock_session()
        redis_client = make_mock_redis()

        job = FakeJob(
            retry_count=rc,
            max_retries=max_retries,
            retry_backoff_base=backoff_base,
        )

        await handle_job_failure(job, "transient error", session, redis_client)

        # PROPERTY: Job must be in QUEUED state (not DEAD_LETTER)
        assert job.status == JobStatus.QUEUED, (
            f"Expected QUEUED but got {job.status} "
            f"(retry_count was {rc}, max_retries={max_retries})"
        )

        # PROPERTY: Job must be added to retry queue (not DLQ)
        redis_client.zadd.assert_called_once()
        call_args = redis_client.zadd.call_args
        assert call_args[0][0] == RETRY_QUEUE_KEY, (
            f"Expected zadd to retry key '{RETRY_QUEUE_KEY}' but got '{call_args[0][0]}'"
        )

    @given(
        max_retries=st.integers(min_value=1, max_value=100),
        retry_count=st.data(),
        backoff_base=backoff_base_strategy,
    )
    @settings(max_examples=200)
    @pytest.mark.asyncio
    async def test_retries_remaining_clears_worker_assignment(
        self, max_retries, retry_count, backoff_base
    ):
        """Re-queued jobs always have worker_id cleared for fresh assignment.

        **Validates: Requirements 8.3**
        """
        rc = retry_count.draw(st.integers(min_value=0, max_value=max_retries - 1))

        session = make_mock_session()
        redis_client = make_mock_redis()

        job = FakeJob(
            retry_count=rc,
            max_retries=max_retries,
            retry_backoff_base=backoff_base,
        )
        # Ensure worker_id is set before failure
        assert job.worker_id is not None

        await handle_job_failure(job, "transient error", session, redis_client)

        # PROPERTY: worker_id must be cleared
        assert job.worker_id is None
