"""Unit tests for the Scheduler promotion loop.

Tests cover:
- Promoting scheduled jobs when execute_at <= now
- Promoting retry jobs when backoff has elapsed
- Atomic ZREM preventing double promotion
- Batch processing (up to 100 per cycle)
- Start/stop lifecycle
- Status transition SCHEDULED → QUEUED in PostgreSQL
"""

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.core.priority_queue import calculate_queue_score
from src.models.base import Base
from src.models.enums import JobStatus
from src.models.job import Job
from src.scheduler.scheduler import Scheduler, SCHEDULED_QUEUE_KEY, RETRY_QUEUE_KEY


@pytest.fixture
async def db_engine():
    """Create an async SQLite engine for testing."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
async def session_factory(db_engine):
    """Create async session factory bound to test engine."""
    factory = async_sessionmaker(
        db_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    return factory


@pytest.fixture
def mock_redis():
    """Create a mock async Redis client."""
    client = AsyncMock()
    client.zrangebyscore = AsyncMock(return_value=[])
    client.zrem = AsyncMock(return_value=1)
    client.zadd = AsyncMock(return_value=1)
    return client


@pytest.fixture
def mock_priority_queue():
    """Create a mock priority queue."""
    queue = AsyncMock()
    queue.enqueue = AsyncMock(return_value=None)
    return queue


@pytest.fixture
def scheduler(session_factory, mock_redis, mock_priority_queue):
    """Create a Scheduler with test dependencies."""
    return Scheduler(
        session_factory=session_factory,
        redis_client=mock_redis,
        priority_queue=mock_priority_queue,
    )


async def _create_scheduled_job(session_factory, job_id=None, priority=5, execute_at=None):
    """Helper to create a scheduled job in the test database."""
    if job_id is None:
        job_id = uuid.uuid4()
    if execute_at is None:
        execute_at = datetime.now(timezone.utc) - timedelta(seconds=10)

    async with session_factory() as session:
        async with session.begin():
            job = Job(
                id=job_id,
                job_type="test_job",
                payload={"key": "value"},
                priority=priority,
                status=JobStatus.SCHEDULED,
                execute_at=execute_at,
                max_retries=3,
                retry_count=0,
                retry_backoff_base=2.0,
                timeout_seconds=300,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
            session.add(job)
    return job_id


async def _create_retry_job(session_factory, job_id=None, priority=3, retry_count=1):
    """Helper to create a retry job (QUEUED status) in the test database."""
    if job_id is None:
        job_id = uuid.uuid4()

    async with session_factory() as session:
        async with session.begin():
            job = Job(
                id=job_id,
                job_type="test_job",
                payload={"key": "value"},
                priority=priority,
                status=JobStatus.QUEUED,
                execute_at=None,
                max_retries=3,
                retry_count=retry_count,
                retry_backoff_base=2.0,
                timeout_seconds=300,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
            session.add(job)
    return job_id


async def _get_job_status(session_factory, job_id):
    """Helper to fetch job status from database."""
    async with session_factory() as session:
        from sqlalchemy import select
        stmt = select(Job).where(Job.id == job_id)
        result = await session.execute(stmt)
        job = result.scalar_one_or_none()
        return job.status if job else None


class TestSchedulerPromoteScheduledJobs:
    """Tests for _promote_scheduled_jobs."""

    async def test_promotes_due_scheduled_job(self, scheduler, session_factory, mock_redis):
        """Req 5.2: Promote when execute_at <= now, set status to QUEUED."""
        job_id = await _create_scheduled_job(session_factory)
        job_id_str = str(job_id)

        # Redis returns this job as due
        mock_redis.zrangebyscore = AsyncMock(return_value=[job_id_str])
        mock_redis.zrem = AsyncMock(return_value=1)

        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        promoted = await scheduler._promote_scheduled_jobs(now_ms)

        assert promoted == 1

        # Verify ZREM was called on the scheduled queue
        mock_redis.zrem.assert_called_once_with(SCHEDULED_QUEUE_KEY, job_id_str)

        # Verify ZADD was called on the priority queue
        mock_redis.zadd.assert_called_once()
        call_args = mock_redis.zadd.call_args
        assert call_args[0][0] == "queue:priority"
        assert job_id_str in call_args[0][1]

        # Verify job status changed to QUEUED in PostgreSQL
        status = await _get_job_status(session_factory, job_id)
        assert status == JobStatus.QUEUED

    async def test_skips_when_zrem_returns_zero(self, scheduler, session_factory, mock_redis):
        """Req 5.6: If ZREM returns 0, skip promotion (already promoted)."""
        job_id = await _create_scheduled_job(session_factory)
        job_id_str = str(job_id)

        # Redis returns job as due but ZREM returns 0 (already removed)
        mock_redis.zrangebyscore = AsyncMock(return_value=[job_id_str])
        mock_redis.zrem = AsyncMock(return_value=0)

        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        promoted = await scheduler._promote_scheduled_jobs(now_ms)

        assert promoted == 0

        # ZADD should NOT have been called
        mock_redis.zadd.assert_not_called()

        # Job status should remain SCHEDULED
        status = await _get_job_status(session_factory, job_id)
        assert status == JobStatus.SCHEDULED

    async def test_promotes_multiple_scheduled_jobs(
        self, scheduler, session_factory, mock_redis
    ):
        """Req 5.3: Process batch of jobs."""
        job_id_1 = await _create_scheduled_job(session_factory, priority=5)
        job_id_2 = await _create_scheduled_job(session_factory, priority=10)

        job_ids = [str(job_id_1), str(job_id_2)]
        mock_redis.zrangebyscore = AsyncMock(return_value=job_ids)
        mock_redis.zrem = AsyncMock(return_value=1)

        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        promoted = await scheduler._promote_scheduled_jobs(now_ms)

        assert promoted == 2

    async def test_returns_zero_when_no_due_jobs(self, scheduler, mock_redis):
        """Req 5.1: No promotion when no jobs are due."""
        mock_redis.zrangebyscore = AsyncMock(return_value=[])

        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        promoted = await scheduler._promote_scheduled_jobs(now_ms)

        assert promoted == 0
        mock_redis.zrem.assert_not_called()
        mock_redis.zadd.assert_not_called()

    async def test_handles_missing_job_in_database(
        self, scheduler, session_factory, mock_redis
    ):
        """Job removed from DB but still in Redis should be skipped gracefully."""
        fake_id = str(uuid.uuid4())
        mock_redis.zrangebyscore = AsyncMock(return_value=[fake_id])
        mock_redis.zrem = AsyncMock(return_value=1)

        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        promoted = await scheduler._promote_scheduled_jobs(now_ms)

        assert promoted == 0
        mock_redis.zadd.assert_not_called()


class TestSchedulerPromoteRetryJobs:
    """Tests for _promote_retry_jobs."""

    async def test_promotes_due_retry_job(self, scheduler, session_factory, mock_redis):
        """Req 5.4: Promote retry jobs when backoff elapses."""
        job_id = await _create_retry_job(session_factory)
        job_id_str = str(job_id)

        mock_redis.zrangebyscore = AsyncMock(return_value=[job_id_str])
        mock_redis.zrem = AsyncMock(return_value=1)

        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        promoted = await scheduler._promote_retry_jobs(now_ms)

        assert promoted == 1

        # Verify ZREM was called on the retry queue
        mock_redis.zrem.assert_called_once_with(RETRY_QUEUE_KEY, job_id_str)

        # Verify ZADD was called on the priority queue
        mock_redis.zadd.assert_called_once()
        call_args = mock_redis.zadd.call_args
        assert call_args[0][0] == "queue:priority"
        assert job_id_str in call_args[0][1]

    async def test_skips_retry_when_zrem_returns_zero(
        self, scheduler, session_factory, mock_redis
    ):
        """Req 5.6: If ZREM returns 0 for retry job, skip."""
        job_id = await _create_retry_job(session_factory)
        job_id_str = str(job_id)

        mock_redis.zrangebyscore = AsyncMock(return_value=[job_id_str])
        mock_redis.zrem = AsyncMock(return_value=0)

        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        promoted = await scheduler._promote_retry_jobs(now_ms)

        assert promoted == 0
        mock_redis.zadd.assert_not_called()

    async def test_returns_zero_when_no_retry_jobs_due(self, scheduler, mock_redis):
        """No retry jobs due returns 0."""
        mock_redis.zrangebyscore = AsyncMock(return_value=[])

        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        promoted = await scheduler._promote_retry_jobs(now_ms)

        assert promoted == 0

    async def test_handles_missing_retry_job_in_database(
        self, scheduler, session_factory, mock_redis
    ):
        """Retry job not found in DB should be skipped gracefully."""
        fake_id = str(uuid.uuid4())
        mock_redis.zrangebyscore = AsyncMock(return_value=[fake_id])
        mock_redis.zrem = AsyncMock(return_value=1)

        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        promoted = await scheduler._promote_retry_jobs(now_ms)

        assert promoted == 0
        mock_redis.zadd.assert_not_called()


class TestSchedulerPromoteDueJobs:
    """Tests for promote_due_jobs (combines scheduled + retry)."""

    async def test_combines_scheduled_and_retry(
        self, scheduler, session_factory, mock_redis
    ):
        """promote_due_jobs combines counts from both queues."""
        scheduled_id = await _create_scheduled_job(session_factory, priority=5)
        retry_id = await _create_retry_job(session_factory, priority=3)

        # First call for scheduled, second for retry
        mock_redis.zrangebyscore = AsyncMock(
            side_effect=[
                [str(scheduled_id)],  # scheduled query
                [str(retry_id)],      # retry query
            ]
        )
        mock_redis.zrem = AsyncMock(return_value=1)

        total = await scheduler.promote_due_jobs()
        assert total == 2

    async def test_returns_zero_when_nothing_due(self, scheduler, mock_redis):
        """Returns 0 when both queues are empty."""
        mock_redis.zrangebyscore = AsyncMock(return_value=[])

        total = await scheduler.promote_due_jobs()
        assert total == 0


class TestSchedulerLifecycle:
    """Tests for start/stop lifecycle."""

    async def test_start_creates_task(self, scheduler):
        """start() should create a background task."""
        await scheduler.start()
        assert scheduler._task is not None
        assert not scheduler._task.done()
        await scheduler.stop()

    async def test_stop_sets_shutdown_flag(self, scheduler):
        """stop() should set the shutdown flag and cancel the task."""
        await scheduler.start()
        await scheduler.stop()
        assert scheduler._shutdown_requested is True
        assert scheduler._task is None

    async def test_scheduler_loop_calls_promote(self, scheduler, mock_redis):
        """The scheduler loop should call promote_due_jobs periodically."""
        mock_redis.zrangebyscore = AsyncMock(return_value=[])

        await scheduler.start()
        # Give the loop time to run at least once
        await asyncio.sleep(0.1)
        await scheduler.stop()

        # Verify zrangebyscore was called (scheduled + retry = at least 2 calls)
        assert mock_redis.zrangebyscore.call_count >= 2


class TestSchedulerBatchSize:
    """Tests for batch size configuration."""

    async def test_uses_configured_batch_size(self, scheduler, mock_redis):
        """Req 5.3: Should query with LIMIT of batch_size (100)."""
        mock_redis.zrangebyscore = AsyncMock(return_value=[])

        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        await scheduler._promote_scheduled_jobs(now_ms)

        # Check that zrangebyscore was called with correct parameters
        mock_redis.zrangebyscore.assert_called_once_with(
            SCHEDULED_QUEUE_KEY,
            min=0,
            max=now_ms,
            start=0,
            num=100,  # settings.scheduler_batch_size default
        )


class TestSchedulerAtomicPromotion:
    """Tests for atomic promotion (Req 5.5)."""

    async def test_zrem_before_zadd_for_scheduled(
        self, scheduler, session_factory, mock_redis
    ):
        """Req 5.5: ZREM must succeed before ZADD is executed."""
        job_id = await _create_scheduled_job(session_factory)
        job_id_str = str(job_id)

        mock_redis.zrangebyscore = AsyncMock(return_value=[job_id_str])

        # Track call order
        call_order = []
        original_zrem = mock_redis.zrem

        async def track_zrem(*args, **kwargs):
            call_order.append("zrem")
            return 1

        async def track_zadd(*args, **kwargs):
            call_order.append("zadd")
            return 1

        mock_redis.zrem = AsyncMock(side_effect=track_zrem)
        mock_redis.zadd = AsyncMock(side_effect=track_zadd)

        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        await scheduler._promote_scheduled_jobs(now_ms)

        # ZREM must be called before ZADD
        assert call_order == ["zrem", "zadd"]

    async def test_partial_batch_promotion(
        self, scheduler, session_factory, mock_redis
    ):
        """If some jobs are already promoted by another instance, only new ones succeed."""
        job_id_1 = await _create_scheduled_job(session_factory, priority=5)
        job_id_2 = await _create_scheduled_job(session_factory, priority=10)

        mock_redis.zrangebyscore = AsyncMock(
            return_value=[str(job_id_1), str(job_id_2)]
        )

        # First ZREM returns 0 (already promoted), second returns 1
        mock_redis.zrem = AsyncMock(side_effect=[0, 1])

        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        promoted = await scheduler._promote_scheduled_jobs(now_ms)

        assert promoted == 1
        # Only one ZADD call
        assert mock_redis.zadd.call_count == 1
