"""Tests for failure detector: dead worker detection and job recovery.

Uses SQLite in-memory for PostgreSQL and a FakeRedis mock for Redis operations.

Requirements covered:
- 8.1: Detect worker with expired heartbeat, mark as DEAD
- 8.2: Recover ALL RUNNING jobs from dead worker
- 8.3: Re-queue jobs with retries remaining (increment retry_count)
- 8.4: Move jobs with exhausted retries to DLQ
- 8.5: Check every 5 seconds
- 8.6: Detection within 20 seconds (15s TTL + 5s check)
- 8.7: Only recover RUNNING jobs (prevents double-recovery)
"""

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Dict, Optional, Set
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.core.failure_detector import DLQ_KEY, FailureDetector
from src.core.priority_queue import PriorityQueueInterface
from src.models.base import Base
from src.models.enums import JobStatus, WorkerStatus
from src.models.job import Job
from src.models.worker import Worker


class FakeRedis:
    """Minimal fake Redis client for testing failure detector logic."""

    def __init__(self):
        self._data: Dict[str, str] = {}
        self._sorted_sets: Dict[str, Dict[str, float]] = {}

    async def get(self, key: str) -> Optional[str]:
        return self._data.get(key)

    async def set(self, key: str, value: str, ex: Optional[int] = None, nx: bool = False):
        if nx and key in self._data:
            return None
        self._data[key] = value
        return True

    async def delete(self, *keys: str) -> int:
        count = 0
        for key in keys:
            if key in self._data:
                del self._data[key]
                count += 1
        return count

    async def exists(self, key: str) -> int:
        return 1 if key in self._data else 0

    async def zadd(self, key: str, mapping: Dict[str, float]) -> int:
        if key not in self._sorted_sets:
            self._sorted_sets[key] = {}
        count = 0
        for member, score in mapping.items():
            if member not in self._sorted_sets[key]:
                count += 1
            self._sorted_sets[key][member] = score
        return count

    async def zscore(self, key: str, member: str) -> Optional[float]:
        if key in self._sorted_sets:
            return self._sorted_sets[key].get(member)
        return None

    async def zrangebyscore(self, key: str, min=None, max=None, start=0, num=100):
        if key not in self._sorted_sets:
            return []
        members = sorted(self._sorted_sets[key].items(), key=lambda x: x[1])
        result = []
        for member, score in members:
            if (min is None or score >= min) and (max is None or score <= max):
                result.append(member)
        return result[start:start + num]

    async def keys(self, pattern: str):
        import fnmatch
        return [k for k in self._data.keys() if fnmatch.fnmatch(k, pattern)]

    async def ping(self):
        return True

    async def aclose(self):
        pass


@pytest.fixture
async def async_engine():
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
async def session_factory(async_engine):
    """Create an async session factory for testing."""
    factory = async_sessionmaker(
        async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    return factory


@pytest.fixture
def fake_redis():
    """Create a FakeRedis instance for testing."""
    return FakeRedis()


@pytest.fixture
def failure_detector(session_factory, fake_redis):
    """Create a FailureDetector instance for testing."""
    mock_pq = AsyncMock(spec=PriorityQueueInterface)
    return FailureDetector(
        session_factory=session_factory,
        redis_client=fake_redis,
        priority_queue=mock_pq,
    )


def make_worker(
    worker_id=None, status=WorkerStatus.ACTIVE, hostname="test-host", pid=1234
):
    """Helper to create a Worker instance."""
    return Worker(
        id=worker_id or uuid.uuid4(),
        hostname=hostname,
        pid=pid,
        status=status,
        last_heartbeat=datetime.now(timezone.utc),
        started_at=datetime.now(timezone.utc),
    )


def make_job(
    job_id=None,
    worker_id=None,
    status=JobStatus.RUNNING,
    retry_count=0,
    max_retries=3,
    priority=5,
):
    """Helper to create a Job instance."""
    now = datetime.now(timezone.utc)
    return Job(
        id=job_id or uuid.uuid4(),
        job_type="test_job",
        payload={"data": "test"},
        priority=priority,
        status=status,
        max_retries=max_retries,
        retry_count=retry_count,
        timeout_seconds=300,
        worker_id=worker_id,
        started_at=now,
        created_at=now,
        updated_at=now,
    )


class TestDetectDeadWorkers:
    """Test dead worker detection (Req 8.1, 8.5, 8.6)."""

    async def test_detect_worker_with_expired_heartbeat(
        self, failure_detector, session_factory, fake_redis
    ):
        """Worker with no heartbeat key in Redis should be detected as dead (Req 8.1)."""
        worker = make_worker()

        async with session_factory() as session:
            async with session.begin():
                session.add(worker)

        # No heartbeat key set → heartbeat expired
        dead_workers = await failure_detector.detect_dead_workers()

        assert worker.id in dead_workers

    async def test_detect_worker_with_valid_heartbeat(
        self, failure_detector, session_factory, fake_redis
    ):
        """Worker with valid heartbeat key should NOT be detected as dead."""
        worker = make_worker()

        async with session_factory() as session:
            async with session.begin():
                session.add(worker)

        # Set heartbeat key
        await fake_redis.set(f"heartbeat:{worker.id}", "alive", ex=15)

        dead_workers = await failure_detector.detect_dead_workers()

        assert worker.id not in dead_workers

    async def test_detect_only_active_workers(
        self, failure_detector, session_factory, fake_redis
    ):
        """Only ACTIVE workers should be checked for heartbeat expiry."""
        active_worker = make_worker(status=WorkerStatus.ACTIVE)
        dead_worker = make_worker(status=WorkerStatus.DEAD)
        idle_worker = make_worker(status=WorkerStatus.IDLE)

        async with session_factory() as session:
            async with session.begin():
                session.add_all([active_worker, dead_worker, idle_worker])

        # No heartbeats set for any worker
        dead_workers = await failure_detector.detect_dead_workers()

        # Only the ACTIVE worker without heartbeat should be detected
        assert active_worker.id in dead_workers
        assert dead_worker.id not in dead_workers
        assert idle_worker.id not in dead_workers

    async def test_detect_multiple_dead_workers(
        self, failure_detector, session_factory, fake_redis
    ):
        """Multiple dead workers should all be detected."""
        worker1 = make_worker()
        worker2 = make_worker()
        worker3 = make_worker()

        async with session_factory() as session:
            async with session.begin():
                session.add_all([worker1, worker2, worker3])

        # Only worker2 has heartbeat
        await fake_redis.set(f"heartbeat:{worker2.id}", "alive", ex=15)

        dead_workers = await failure_detector.detect_dead_workers()

        assert worker1.id in dead_workers
        assert worker2.id not in dead_workers
        assert worker3.id in dead_workers

    async def test_no_active_workers_returns_empty(
        self, failure_detector, session_factory
    ):
        """No active workers means no dead workers detected."""
        dead_workers = await failure_detector.detect_dead_workers()
        assert dead_workers == []


class TestRecoverAbandonedJobs:
    """Test job recovery from dead workers (Req 8.2, 8.3, 8.4, 8.7)."""

    async def test_requeue_job_with_retries_remaining(
        self, failure_detector, session_factory, fake_redis
    ):
        """Jobs with retries remaining should be re-queued (Req 8.3)."""
        worker = make_worker()
        job = make_job(worker_id=worker.id, retry_count=0, max_retries=3)

        async with session_factory() as session:
            async with session.begin():
                session.add(worker)
                session.add(job)

        recovered = await failure_detector.recover_abandoned_jobs(worker.id)

        assert recovered == 1

        # Verify job state in database
        async with session_factory() as session:
            result = await session.execute(select(Job).where(Job.id == job.id))
            updated_job = result.scalar_one()

            assert updated_job.status == JobStatus.QUEUED
            assert updated_job.retry_count == 1
            assert updated_job.worker_id is None
            assert f"Worker {worker.id} died" in updated_job.error

    async def test_requeue_adds_to_priority_queue_in_redis(
        self, failure_detector, session_factory, fake_redis
    ):
        """Re-queued jobs should be added to the Redis priority queue."""
        worker = make_worker()
        job = make_job(worker_id=worker.id, retry_count=0, max_retries=3, priority=7)

        async with session_factory() as session:
            async with session.begin():
                session.add(worker)
                session.add(job)

        await failure_detector.recover_abandoned_jobs(worker.id)

        # Verify job was added to priority queue in Redis
        score = await fake_redis.zscore("queue:priority", str(job.id))
        assert score is not None

    async def test_dlq_job_with_exhausted_retries(
        self, failure_detector, session_factory, fake_redis
    ):
        """Jobs with exhausted retries should go to DLQ (Req 8.4)."""
        worker = make_worker()
        job = make_job(worker_id=worker.id, retry_count=3, max_retries=3)

        async with session_factory() as session:
            async with session.begin():
                session.add(worker)
                session.add(job)

        recovered = await failure_detector.recover_abandoned_jobs(worker.id)

        assert recovered == 1

        # Verify job state in database
        async with session_factory() as session:
            result = await session.execute(select(Job).where(Job.id == job.id))
            updated_job = result.scalar_one()

            assert updated_job.status == JobStatus.DEAD_LETTER
            assert updated_job.worker_id is None
            assert "max retries" in updated_job.error

    async def test_dlq_adds_to_dlq_sorted_set_in_redis(
        self, failure_detector, session_factory, fake_redis
    ):
        """DLQ jobs should be added to the Redis DLQ sorted set."""
        worker = make_worker()
        job = make_job(worker_id=worker.id, retry_count=3, max_retries=3)

        async with session_factory() as session:
            async with session.begin():
                session.add(worker)
                session.add(job)

        await failure_detector.recover_abandoned_jobs(worker.id)

        # Verify job was added to DLQ in Redis
        score = await fake_redis.zscore(DLQ_KEY, str(job.id))
        assert score is not None

    async def test_recover_all_running_jobs(
        self, failure_detector, session_factory, fake_redis
    ):
        """ALL RUNNING jobs from dead worker should be recovered (Req 8.2)."""
        worker = make_worker()
        job1 = make_job(worker_id=worker.id, retry_count=0, max_retries=3)
        job2 = make_job(worker_id=worker.id, retry_count=1, max_retries=3)
        job3 = make_job(worker_id=worker.id, retry_count=3, max_retries=3)

        async with session_factory() as session:
            async with session.begin():
                session.add(worker)
                session.add_all([job1, job2, job3])

        recovered = await failure_detector.recover_abandoned_jobs(worker.id)

        assert recovered == 3

        # Verify each job's state
        async with session_factory() as session:
            r1 = await session.execute(select(Job).where(Job.id == job1.id))
            r2 = await session.execute(select(Job).where(Job.id == job2.id))
            r3 = await session.execute(select(Job).where(Job.id == job3.id))

            assert r1.scalar_one().status == JobStatus.QUEUED
            assert r2.scalar_one().status == JobStatus.QUEUED
            assert r3.scalar_one().status == JobStatus.DEAD_LETTER

    async def test_only_recover_running_jobs(
        self, failure_detector, session_factory, fake_redis
    ):
        """Only RUNNING jobs should be recovered - not QUEUED/COMPLETED (Req 8.7)."""
        worker = make_worker()
        running_job = make_job(
            worker_id=worker.id, status=JobStatus.RUNNING, retry_count=0
        )
        queued_job = make_job(
            worker_id=worker.id, status=JobStatus.QUEUED, retry_count=0
        )
        completed_job = make_job(
            worker_id=worker.id, status=JobStatus.COMPLETED, retry_count=0
        )

        async with session_factory() as session:
            async with session.begin():
                session.add(worker)
                session.add_all([running_job, queued_job, completed_job])

        recovered = await failure_detector.recover_abandoned_jobs(worker.id)

        # Only the RUNNING job should be recovered
        assert recovered == 1

        async with session_factory() as session:
            r_running = await session.execute(
                select(Job).where(Job.id == running_job.id)
            )
            r_queued = await session.execute(
                select(Job).where(Job.id == queued_job.id)
            )
            r_completed = await session.execute(
                select(Job).where(Job.id == completed_job.id)
            )

            assert r_running.scalar_one().status == JobStatus.QUEUED
            assert r_queued.scalar_one().status == JobStatus.QUEUED  # Unchanged
            assert r_completed.scalar_one().status == JobStatus.COMPLETED  # Unchanged

    async def test_mark_worker_as_dead(
        self, failure_detector, session_factory, fake_redis
    ):
        """Worker should be marked as DEAD in PostgreSQL (Req 8.1)."""
        worker = make_worker()
        job = make_job(worker_id=worker.id)

        async with session_factory() as session:
            async with session.begin():
                session.add(worker)
                session.add(job)

        await failure_detector.recover_abandoned_jobs(worker.id)

        # Verify worker is marked DEAD
        async with session_factory() as session:
            result = await session.execute(
                select(Worker).where(Worker.id == worker.id)
            )
            updated_worker = result.scalar_one()
            assert updated_worker.status == WorkerStatus.DEAD

    async def test_delete_heartbeat_key(
        self, failure_detector, session_factory, fake_redis
    ):
        """Heartbeat key should be deleted from Redis after recovery (Req 8.1)."""
        worker = make_worker()
        job = make_job(worker_id=worker.id)

        async with session_factory() as session:
            async with session.begin():
                session.add(worker)
                session.add(job)

        # Set a heartbeat key (simulating a stale key)
        await fake_redis.set(f"heartbeat:{worker.id}", "stale")

        await failure_detector.recover_abandoned_jobs(worker.id)

        # Verify heartbeat key is deleted
        exists = await fake_redis.exists(f"heartbeat:{worker.id}")
        assert exists == 0

    async def test_no_running_jobs_returns_zero(
        self, failure_detector, session_factory, fake_redis
    ):
        """If dead worker has no RUNNING jobs, return 0."""
        worker = make_worker()

        async with session_factory() as session:
            async with session.begin():
                session.add(worker)

        recovered = await failure_detector.recover_abandoned_jobs(worker.id)
        assert recovered == 0

    async def test_retry_count_increment(
        self, failure_detector, session_factory, fake_redis
    ):
        """Retry count should be incremented on re-queue (Req 8.3)."""
        worker = make_worker()
        job = make_job(worker_id=worker.id, retry_count=2, max_retries=3)

        async with session_factory() as session:
            async with session.begin():
                session.add(worker)
                session.add(job)

        await failure_detector.recover_abandoned_jobs(worker.id)

        async with session_factory() as session:
            result = await session.execute(select(Job).where(Job.id == job.id))
            updated_job = result.scalar_one()
            assert updated_job.retry_count == 3  # Was 2, now 3

    async def test_boundary_retry_count_equals_max(
        self, failure_detector, session_factory, fake_redis
    ):
        """When retry_count equals max_retries, job goes to DLQ (Req 8.4)."""
        worker = make_worker()
        # retry_count == max_retries → exhausted
        job = make_job(worker_id=worker.id, retry_count=3, max_retries=3)

        async with session_factory() as session:
            async with session.begin():
                session.add(worker)
                session.add(job)

        await failure_detector.recover_abandoned_jobs(worker.id)

        async with session_factory() as session:
            result = await session.execute(select(Job).where(Job.id == job.id))
            updated_job = result.scalar_one()
            assert updated_job.status == JobStatus.DEAD_LETTER

    async def test_boundary_retry_count_one_less_than_max(
        self, failure_detector, session_factory, fake_redis
    ):
        """When retry_count is one less than max_retries, job is re-queued."""
        worker = make_worker()
        # retry_count < max_retries → re-queue
        job = make_job(worker_id=worker.id, retry_count=2, max_retries=3)

        async with session_factory() as session:
            async with session.begin():
                session.add(worker)
                session.add(job)

        await failure_detector.recover_abandoned_jobs(worker.id)

        async with session_factory() as session:
            result = await session.execute(select(Job).where(Job.id == job.id))
            updated_job = result.scalar_one()
            assert updated_job.status == JobStatus.QUEUED
            assert updated_job.retry_count == 3


class TestFailureDetectorLifecycle:
    """Test start/stop lifecycle of the failure detector."""

    async def test_start_creates_background_task(
        self, failure_detector
    ):
        """start() should create a background task."""
        await failure_detector.start()
        assert failure_detector._task is not None
        assert not failure_detector._task.done()
        await failure_detector.stop()

    async def test_stop_cancels_background_task(
        self, failure_detector
    ):
        """stop() should cancel the background task."""
        await failure_detector.start()
        await failure_detector.stop()
        assert failure_detector._task is None

    async def test_stop_is_idempotent(self, failure_detector):
        """Calling stop() when not started should not raise."""
        await failure_detector.stop()  # Should not raise

    async def test_detection_loop_interval(
        self, failure_detector, session_factory, fake_redis
    ):
        """Failure detector should check periodically (Req 8.5)."""
        # Patch settings to use a very short interval for testing
        with patch("src.core.failure_detector.settings") as mock_settings:
            mock_settings.failure_check_interval_seconds = 0.1

            call_count = 0
            original_detect = failure_detector.detect_dead_workers

            async def counting_detect():
                nonlocal call_count
                call_count += 1
                return []

            failure_detector.detect_dead_workers = counting_detect

            await failure_detector.start()
            await asyncio.sleep(0.35)
            await failure_detector.stop()

            # Should have been called multiple times in 0.35s with 0.1s interval
            assert call_count >= 2


class TestFailureDetectorErrorHandling:
    """Test error handling in the failure detector loop."""

    async def test_loop_continues_on_error(
        self, failure_detector, session_factory, fake_redis
    ):
        """Loop should continue running after an exception."""
        call_count = 0

        async def failing_detect():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Simulated error")
            return []

        with patch("src.core.failure_detector.settings") as mock_settings:
            mock_settings.failure_check_interval_seconds = 0.05

            failure_detector.detect_dead_workers = failing_detect

            await failure_detector.start()
            await asyncio.sleep(0.2)
            await failure_detector.stop()

            # Should have been called more than once despite error
            assert call_count >= 2
