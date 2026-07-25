"""Property-based tests for dead-letter retry reset.

**Property 17: Dead-Letter Retry Reset**

**Validates: Requirements 10.3, 10.4**

Uses Hypothesis to verify that retrying DLQ jobs:
- Always resets retry_count to 0 and transitions status to QUEUED (Req 10.3)
- Preserves the original payload after retry (Req 10.3, 10.5)
- Returns 409 for any non-DEAD_LETTER job (Req 10.4)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from src.core.coordinator import JobCoordinator, DLQ_KEY
from src.models.enums import JobStatus


# --- Strategies ---

# Strategy for retry_count: any non-negative integer a job might have accumulated
retry_count_strategy = st.integers(min_value=0, max_value=100)

# Strategy for priority values
priority_strategy = st.integers(min_value=0, max_value=10000)

# Strategy for JSON-serializable payloads
json_primitives = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-1000000, max_value=1000000),
    st.floats(allow_nan=False, allow_infinity=False),
    st.text(min_size=0, max_size=50),
)

json_payload_strategy = st.dictionaries(
    keys=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("L", "N"))),
    values=json_primitives,
    min_size=1,
    max_size=10,
)

# Strategy for job_type strings
job_type_strategy = st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("L", "N", "Pd")))

# Strategy for non-DEAD_LETTER statuses (for the guard property)
non_dead_letter_status_strategy = st.sampled_from(
    [s for s in JobStatus if s != JobStatus.DEAD_LETTER]
)

# Strategy for error messages
error_strategy = st.text(min_size=1, max_size=200)


# --- FakeJob dataclass ---

@dataclass
class FakeJob:
    """Lightweight job stand-in for coordinator DLQ retry property tests."""

    id: UUID = field(default_factory=uuid4)
    job_type: str = "test_job"
    payload: dict = field(default_factory=lambda: {"key": "value"})
    priority: int = 5
    status: JobStatus = JobStatus.DEAD_LETTER
    error: Optional[str] = "max retries exceeded"
    retry_count: int = 3
    max_retries: int = 3
    retry_backoff_base: float = 2.0
    timeout_seconds: int = 300
    worker_id: Optional[UUID] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[dict] = None
    execute_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


# --- Helpers ---

def _make_mock_session_factory(job):
    """Create a mock session factory that returns the given job (or None if job is None)."""
    mock_session = AsyncMock()
    mock_result = MagicMock()

    mock_result.scalar_one_or_none.return_value = job

    mock_session.execute = AsyncMock(return_value=mock_result)
    mock_session.flush = AsyncMock()
    mock_session.add = MagicMock()

    # Make session.begin() work as async context manager
    mock_begin = AsyncMock()
    mock_begin.__aenter__ = AsyncMock(return_value=None)
    mock_begin.__aexit__ = AsyncMock(return_value=None)
    mock_session.begin = MagicMock(return_value=mock_begin)

    # Session factory as async context manager
    mock_factory = MagicMock()
    mock_ctx = AsyncMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
    mock_ctx.__aexit__ = AsyncMock(return_value=None)
    mock_factory.return_value = mock_ctx

    return mock_factory


def _create_coordinator(mock_session_factory, mock_redis, mock_priority_queue):
    """Helper to create a JobCoordinator with mocked dependencies."""
    return JobCoordinator(
        session_factory=mock_session_factory,
        redis_client=mock_redis,
        priority_queue=mock_priority_queue,
    )


# --- Property Tests ---


class TestDeadLetterRetryReset:
    """Property 17: Dead-Letter Retry Reset.

    **Validates: Requirements 10.3, 10.4**
    """

    @given(
        retry_count=retry_count_strategy,
        priority=priority_strategy,
        payload=json_payload_strategy,
        job_type=job_type_strategy,
        error_msg=error_strategy,
    )
    @settings(max_examples=200)
    async def test_retry_resets_retry_count_and_status_for_any_dead_letter_job(
        self, retry_count, priority, payload, job_type, error_msg
    ):
        """Property 17a: For ANY DEAD_LETTER job with ANY retry_count and ANY payload,
        retry_dlq_job always results in retry_count=0 and status=QUEUED.

        **Validates: Requirements 10.3**
        """
        job = FakeJob(
            status=JobStatus.DEAD_LETTER,
            retry_count=retry_count,
            priority=priority,
            payload=payload,
            job_type=job_type,
            error=error_msg,
        )

        mock_factory = _make_mock_session_factory(job)
        mock_redis = AsyncMock()
        mock_redis.zrem = AsyncMock()
        mock_pq = AsyncMock()
        mock_pq.enqueue = AsyncMock()

        coordinator = _create_coordinator(mock_factory, mock_redis, mock_pq)
        result = await coordinator.retry_dlq_job(job.id)

        # retry_count must be reset to 0
        assert result["retry_count"] == 0
        assert job.retry_count == 0

        # status must be QUEUED
        assert result["status"] == "QUEUED"
        assert job.status == JobStatus.QUEUED

        # status_code must be 200
        assert result["status_code"] == 200

    @given(
        payload=json_payload_strategy,
        retry_count=retry_count_strategy,
        priority=priority_strategy,
        job_type=job_type_strategy,
    )
    @settings(max_examples=200)
    async def test_retry_preserves_original_payload(
        self, payload, retry_count, priority, job_type
    ):
        """Property 17b: For ANY DEAD_LETTER job, the original payload is preserved after retry.

        **Validates: Requirements 10.3**
        """
        job = FakeJob(
            status=JobStatus.DEAD_LETTER,
            payload=payload,
            retry_count=retry_count,
            priority=priority,
            job_type=job_type,
        )

        mock_factory = _make_mock_session_factory(job)
        mock_redis = AsyncMock()
        mock_redis.zrem = AsyncMock()
        mock_pq = AsyncMock()
        mock_pq.enqueue = AsyncMock()

        coordinator = _create_coordinator(mock_factory, mock_redis, mock_pq)
        result = await coordinator.retry_dlq_job(job.id)

        # Original payload must be preserved in the response
        assert result["payload"] == payload
        # The job object's payload must remain unchanged
        assert job.payload == payload

    @given(status=non_dead_letter_status_strategy)
    @settings(max_examples=200)
    async def test_non_dead_letter_jobs_return_409(self, status):
        """Property 17c: For ANY non-DEAD_LETTER job, retry returns 409.

        **Validates: Requirements 10.4**
        """
        job = FakeJob(status=status)

        mock_factory = _make_mock_session_factory(job)
        mock_redis = AsyncMock()
        mock_pq = AsyncMock()

        coordinator = _create_coordinator(mock_factory, mock_redis, mock_pq)
        result = await coordinator.retry_dlq_job(job.id)

        # Must return 409 for non-DEAD_LETTER jobs
        assert result["status_code"] == 409
        assert result["error"] == "not_retryable"

        # Job status must NOT be modified
        assert job.status == status
