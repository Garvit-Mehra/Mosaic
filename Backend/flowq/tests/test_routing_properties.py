"""Property-based tests for job routing correctness.

**Validates: Requirements 1.2, 1.3**

Uses Hypothesis to generate valid job submissions with and without execute_at,
verifying that:
- Jobs without execute_at are routed to QUEUED status and added to the priority queue.
- Jobs with future execute_at are routed to SCHEDULED status and added to the schedule set.
"""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

from hypothesis import given, settings
from hypothesis import strategies as st
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.core.coordinator import JobCoordinator, SCHEDULED_QUEUE_KEY
from src.models.base import Base
from src.models.enums import JobStatus


# --- Strategies ---

# Valid job types (we mock out validation, so any non-empty string works)
job_type_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="_-"),
    min_size=1,
    max_size=50,
)

# Valid priorities: 0 to 10000
priority_strategy = st.integers(min_value=0, max_value=10000)

# Valid payloads: simple JSON-serializable dicts
payload_strategy = st.fixed_dictionaries(
    {},
    optional={
        "key": st.text(min_size=0, max_size=20),
        "value": st.integers(min_value=-1000, max_value=1000),
        "flag": st.booleans(),
    },
)

# Future execute_at: 1 minute to 30 days in the future
future_execute_at_strategy = st.integers(
    min_value=60, max_value=30 * 24 * 3600
).map(lambda secs: datetime.now(timezone.utc) + timedelta(seconds=secs))


# --- Helper to create coordinator with fresh DB per test call ---


async def _make_coordinator():
    """Create a fresh coordinator with in-memory SQLite DB and mock Redis/queue."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    mock_redis = AsyncMock()
    mock_redis.zadd = AsyncMock(return_value=1)

    mock_priority_queue = AsyncMock()
    mock_priority_queue.enqueue = AsyncMock(return_value=None)

    coordinator = JobCoordinator(
        session_factory=session_factory,
        redis_client=mock_redis,
        priority_queue=mock_priority_queue,
    )

    return coordinator, mock_redis, mock_priority_queue, engine


# --- Property Tests ---


class TestJobRoutingCorrectness:
    """Property 14: Job Routing Correctness.

    **Validates: Requirements 1.2, 1.3**
    """

    @given(
        job_type=job_type_strategy,
        payload=payload_strategy,
        priority=priority_strategy,
    )
    @settings(max_examples=100)
    async def test_immediate_jobs_are_queued_in_priority_queue(
        self, job_type, payload, priority
    ):
        """Jobs without execute_at transition to QUEUED and are added to the priority queue.

        **Validates: Requirements 1.2**

        Property: For ALL valid job submissions where execute_at is None,
        the resulting job status is QUEUED, priority_queue.enqueue is called,
        and redis.zadd (for schedule set) is NOT called.
        """
        coordinator, mock_redis, mock_priority_queue, engine = await _make_coordinator()

        try:
            with patch("src.core.coordinator.validate_job_submission"):
                result = await coordinator.submit_job(
                    job_type=job_type,
                    payload=payload,
                    priority=priority,
                    execute_at=None,
                )

            # Job must be QUEUED
            assert result["status"] == "QUEUED", (
                f"Expected QUEUED but got {result['status']} for immediate job"
            )

            # priority_queue.enqueue must have been called with correct job_id and priority
            mock_priority_queue.enqueue.assert_called_once()
            call_kwargs = mock_priority_queue.enqueue.call_args.kwargs
            assert call_kwargs["job_id"] == result["id"]
            assert call_kwargs["priority"] == priority

            # redis.zadd (schedule set) must NOT have been called
            mock_redis.zadd.assert_not_called()
        finally:
            await engine.dispose()

    @given(
        job_type=job_type_strategy,
        payload=payload_strategy,
        priority=priority_strategy,
        execute_at=future_execute_at_strategy,
    )
    @settings(max_examples=100)
    async def test_scheduled_jobs_are_placed_in_schedule_set(
        self, job_type, payload, priority, execute_at
    ):
        """Jobs with future execute_at transition to SCHEDULED and are added to the schedule set.

        **Validates: Requirements 1.3**

        Property: For ALL valid job submissions where execute_at is a future datetime,
        the resulting job status is SCHEDULED, redis.zadd is called with the correct
        schedule key and score (execute_at as ms timestamp), and priority_queue.enqueue
        is NOT called.
        """
        coordinator, mock_redis, mock_priority_queue, engine = await _make_coordinator()

        try:
            with patch("src.core.coordinator.validate_job_submission"):
                result = await coordinator.submit_job(
                    job_type=job_type,
                    payload=payload,
                    priority=priority,
                    execute_at=execute_at,
                )

            # Job must be SCHEDULED
            assert result["status"] == "SCHEDULED", (
                f"Expected SCHEDULED but got {result['status']} for job with execute_at"
            )

            # redis.zadd must have been called for the schedule set
            mock_redis.zadd.assert_called_once()
            zadd_args = mock_redis.zadd.call_args
            # First positional arg is the key
            assert zadd_args.args[0] == SCHEDULED_QUEUE_KEY

            # The score should be execute_at as ms timestamp
            mapping = zadd_args.args[1]
            job_id_str = str(result["id"])
            assert job_id_str in mapping

            # Ensure the score is the execute_at timestamp in milliseconds
            if execute_at.tzinfo is None:
                execute_at_aware = execute_at.replace(tzinfo=timezone.utc)
            else:
                execute_at_aware = execute_at
            expected_score = int(execute_at_aware.timestamp() * 1000)
            assert mapping[job_id_str] == expected_score

            # priority_queue.enqueue must NOT have been called
            mock_priority_queue.enqueue.assert_not_called()
        finally:
            await engine.dispose()
