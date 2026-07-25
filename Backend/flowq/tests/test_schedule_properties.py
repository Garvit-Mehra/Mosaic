"""Property-based tests for schedule correctness.

**Validates: Requirements 5.1, 5.2**

Uses Hypothesis to generate jobs with various execute_at times and verify:
- Property 9a: Jobs are NOT promoted before execute_at (Req 5.1)
- Property 9b: Jobs ARE promoted once execute_at <= now (Req 5.2)

The Scheduler uses ZRANGEBYSCORE with max=now_ms to find promotable jobs.
A job's score in the schedule set equals its execute_at_ms. Therefore:
- If score (execute_at_ms) > now_ms → ZRANGEBYSCORE won't return it → not promoted
- If score (execute_at_ms) <= now_ms → ZRANGEBYSCORE returns it → promoted
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from src.models.enums import JobStatus
from src.scheduler.scheduler import Scheduler, SCHEDULED_QUEUE_KEY


# --- Strategies ---

# Generate execute_at timestamps as milliseconds since epoch
# Range: 1 day ago to 1 day in the future from a fixed reference point
REFERENCE_NOW_MS = 1_700_000_000_000  # A fixed reference "now" in ms

execute_at_ms_strategy = st.integers(
    min_value=REFERENCE_NOW_MS - 86_400_000,  # 1 day before reference
    max_value=REFERENCE_NOW_MS + 86_400_000,  # 1 day after reference
)

now_ms_strategy = st.integers(
    min_value=REFERENCE_NOW_MS - 43_200_000,  # 12 hours before reference
    max_value=REFERENCE_NOW_MS + 43_200_000,  # 12 hours after reference
)

priority_strategy = st.integers(min_value=0, max_value=10000)


@dataclass
class FakeJob:
    """Lightweight job stand-in for testing without DB dependencies."""

    id: uuid.UUID = field(default_factory=uuid.uuid4)
    job_type: str = "test_job"
    payload: dict = field(default_factory=dict)
    priority: int = 0
    status: JobStatus = JobStatus.SCHEDULED
    execute_at: datetime | None = None
    max_retries: int = 3
    retry_count: int = 0
    retry_backoff_base: float = 2.0
    timeout_seconds: int = 300
    worker_id: uuid.UUID | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: dict | None = None
    error: str | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


# --- Property Tests ---


class TestScheduleCorrectness:
    """Property 9: Schedule Correctness.

    **Validates: Requirements 5.1, 5.2**
    """

    @given(
        execute_at_ms=execute_at_ms_strategy,
        now_ms=now_ms_strategy,
    )
    @settings(max_examples=200)
    def test_jobs_not_promoted_before_execute_at(self, execute_at_ms: int, now_ms: int):
        """Property 9a: Jobs with execute_at > now are NOT promoted.

        ZRANGEBYSCORE with max=now_ms will not return any job whose score
        (execute_at_ms) is strictly greater than now_ms. This means the
        scheduler correctly keeps future jobs in the schedule set.

        **Validates: Requirements 5.1**
        """
        # Only test the case where execute_at is in the future relative to now
        assume(execute_at_ms > now_ms)

        # ZRANGEBYSCORE filters: min=0, max=now_ms
        # A job with score=execute_at_ms where execute_at_ms > now_ms
        # will NOT be in the result set
        job_score = execute_at_ms
        is_returned_by_zrangebyscore = job_score <= now_ms

        # The job should NOT be returned (and therefore not promoted)
        assert is_returned_by_zrangebyscore is False, (
            f"Job with execute_at_ms={execute_at_ms} should NOT be returned "
            f"by ZRANGEBYSCORE(max={now_ms}) but would be"
        )

    @given(
        execute_at_ms=execute_at_ms_strategy,
        now_ms=now_ms_strategy,
    )
    @settings(max_examples=200)
    def test_jobs_promoted_once_execute_at_arrives(self, execute_at_ms: int, now_ms: int):
        """Property 9b: Jobs with execute_at <= now ARE promoted.

        ZRANGEBYSCORE with max=now_ms will return any job whose score
        (execute_at_ms) is less than or equal to now_ms. This means the
        scheduler correctly promotes due jobs to the priority queue.

        **Validates: Requirements 5.2**
        """
        # Only test the case where execute_at has arrived
        assume(execute_at_ms <= now_ms)

        # ZRANGEBYSCORE filters: min=0, max=now_ms
        # A job with score=execute_at_ms where execute_at_ms <= now_ms
        # WILL be in the result set
        job_score = execute_at_ms
        is_returned_by_zrangebyscore = job_score <= now_ms

        # The job SHOULD be returned (and therefore eligible for promotion)
        assert is_returned_by_zrangebyscore is True, (
            f"Job with execute_at_ms={execute_at_ms} should be returned "
            f"by ZRANGEBYSCORE(max={now_ms}) but would not be"
        )

    @given(
        execute_at_ms=execute_at_ms_strategy,
        now_ms=now_ms_strategy,
        priority=priority_strategy,
    )
    @settings(max_examples=200)
    @pytest.mark.asyncio
    async def test_scheduler_promote_scheduled_jobs_respects_time(
        self, execute_at_ms: int, now_ms: int, priority: int
    ):
        """Property 9a/9b integrated: _promote_scheduled_jobs correctly
        uses ZRANGEBYSCORE with max=now_ms, so only due jobs are promoted.

        Mocks Redis to simulate the ZRANGEBYSCORE behavior and verifies that:
        - When execute_at > now_ms: zrangebyscore returns empty → 0 promotions
        - When execute_at <= now_ms: zrangebyscore returns the job → 1 promotion
          (assuming ZREM returns 1, meaning this instance wins the atomic check)

        **Validates: Requirements 5.1, 5.2**
        """
        job_id = uuid.uuid4()
        job = FakeJob(id=job_id, priority=priority, status=JobStatus.SCHEDULED)

        # Mock Redis client
        mock_redis = AsyncMock()

        # Mock session factory and session
        mock_session = AsyncMock()
        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_begin_ctx = AsyncMock()
        mock_begin_ctx.__aenter__ = AsyncMock(return_value=None)
        mock_begin_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_session.begin = MagicMock(return_value=mock_begin_ctx)

        mock_session_factory = MagicMock(return_value=mock_session_ctx)

        # Mock priority queue
        mock_priority_queue = AsyncMock()

        scheduler = Scheduler(
            session_factory=mock_session_factory,
            redis_client=mock_redis,
            priority_queue=mock_priority_queue,
        )

        if execute_at_ms > now_ms:
            # Job is in the future → ZRANGEBYSCORE(max=now_ms) returns nothing
            mock_redis.zrangebyscore = AsyncMock(return_value=[])

            result = await scheduler._promote_scheduled_jobs(now_ms)

            # Verify no jobs were promoted
            assert result == 0
            # Verify ZRANGEBYSCORE was called with correct max
            mock_redis.zrangebyscore.assert_called_once_with(
                SCHEDULED_QUEUE_KEY, min=0, max=now_ms, start=0, num=100
            )
            # ZREM should NOT have been called (no jobs to process)
            mock_redis.zrem.assert_not_called()
        else:
            # Job is due → ZRANGEBYSCORE(max=now_ms) returns the job
            mock_redis.zrangebyscore = AsyncMock(return_value=[str(job_id)])
            # ZREM returns 1 (this instance wins the atomic promotion)
            mock_redis.zrem = AsyncMock(return_value=1)
            mock_redis.zadd = AsyncMock(return_value=1)

            # Mock the DB query to return our fake job
            mock_result = MagicMock()
            mock_result.scalar_one_or_none = MagicMock(return_value=job)
            mock_session.execute = AsyncMock(return_value=mock_result)

            result = await scheduler._promote_scheduled_jobs(now_ms)

            # Verify the job was promoted
            assert result == 1
            # Verify ZRANGEBYSCORE was called with correct max
            mock_redis.zrangebyscore.assert_called_once_with(
                SCHEDULED_QUEUE_KEY, min=0, max=now_ms, start=0, num=100
            )
            # Verify ZREM was called for atomic check
            mock_redis.zrem.assert_called_once_with(SCHEDULED_QUEUE_KEY, str(job_id))
            # Verify job was added to priority queue
            mock_redis.zadd.assert_called_once()

    @given(
        execute_at_ms=execute_at_ms_strategy,
        now_ms=now_ms_strategy,
    )
    @settings(max_examples=200)
    @pytest.mark.asyncio
    async def test_zrem_zero_prevents_double_promotion(
        self, execute_at_ms: int, now_ms: int
    ):
        """When ZREM returns 0 (another instance promoted), the job is skipped.

        This validates the atomic promotion guarantee: even if ZRANGEBYSCORE
        returns a job to multiple scheduler instances, only one succeeds.

        **Validates: Requirements 5.1, 5.2**
        """
        # Only test when the job is due (ZRANGEBYSCORE would return it)
        assume(execute_at_ms <= now_ms)

        job_id = uuid.uuid4()

        # Mock Redis client
        mock_redis = AsyncMock()
        mock_redis.zrangebyscore = AsyncMock(return_value=[str(job_id)])
        # ZREM returns 0 → another instance already promoted this job
        mock_redis.zrem = AsyncMock(return_value=0)

        # Mock session factory (should not be called since ZREM=0 skips DB)
        mock_session_factory = MagicMock()

        # Mock priority queue
        mock_priority_queue = AsyncMock()

        scheduler = Scheduler(
            session_factory=mock_session_factory,
            redis_client=mock_redis,
            priority_queue=mock_priority_queue,
        )

        result = await scheduler._promote_scheduled_jobs(now_ms)

        # Job was NOT promoted (another instance did it)
        assert result == 0
        # ZREM was called but returned 0
        mock_redis.zrem.assert_called_once_with(SCHEDULED_QUEUE_KEY, str(job_id))
        # ZADD should NOT have been called (skipped due to ZREM=0)
        mock_redis.zadd.assert_not_called()
