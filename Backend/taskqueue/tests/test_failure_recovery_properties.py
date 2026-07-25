"""Property-based tests for failure recovery completeness.

**Validates: Requirements 8.1, 8.2, 8.3, 8.4**

Uses Hypothesis to generate random sets of RUNNING jobs assigned to a dead
worker with varying retry_count/max_retries values, and verifies:
- Property 18a: ALL RUNNING jobs are recovered (recovered count == total RUNNING jobs)
- Property 18b: After recovery, NO job remains in RUNNING status assigned to the dead worker
- Property 18c: Jobs with retry_count < max_retries are ALWAYS re-queued (status=QUEUED)
- Property 18d: Jobs with retry_count >= max_retries are ALWAYS moved to DLQ (status=DEAD_LETTER)
"""

import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pytest
from hypothesis import given, settings as hypothesis_settings, assume
from hypothesis import strategies as st
from sqlalchemy import select
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.core.failure_detector import FailureDetector
from src.core.priority_queue import PriorityQueueInterface
from src.models.base import Base
from src.models.enums import JobStatus, WorkerStatus
from src.models.job import Job
from src.models.worker import Worker


# ---------------------------------------------------------------------------
# Fake Redis (same pattern as test_failure_detector.py)
# ---------------------------------------------------------------------------


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

    async def ping(self):
        return True

    async def aclose(self):
        pass


# ---------------------------------------------------------------------------
# Fake Priority Queue
# ---------------------------------------------------------------------------


class FakePriorityQueue(PriorityQueueInterface):
    """Fake priority queue that records enqueue calls."""

    def __init__(self):
        self.enqueued: List[uuid.UUID] = []

    async def enqueue(self, job_id, priority, enqueued_at):
        self.enqueued.append(job_id)

    async def dequeue(self, timeout=5.0):
        return None

    async def remove(self, job_id):
        return False

    async def depth(self):
        return 0

    async def peek(self, count=10):
        return []


# ---------------------------------------------------------------------------
# Hypothesis Strategies
# ---------------------------------------------------------------------------


@st.composite
def job_config_strategy(draw):
    """Generate a job configuration with random retry_count, max_retries, and priority."""
    max_retries = draw(st.integers(min_value=0, max_value=10))
    retry_count = draw(st.integers(min_value=0, max_value=max_retries + 3))
    priority = draw(st.integers(min_value=0, max_value=100))
    return {
        "retry_count": retry_count,
        "max_retries": max_retries,
        "priority": priority,
    }


jobs_list_strategy = st.lists(
    job_config_strategy(),
    min_size=1,
    max_size=10,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_worker(worker_id=None, status=WorkerStatus.ACTIVE):
    """Create a Worker instance."""
    return Worker(
        id=worker_id or uuid.uuid4(),
        hostname="test-host",
        pid=1234,
        status=status,
        last_heartbeat=datetime.now(timezone.utc),
        started_at=datetime.now(timezone.utc),
    )


def make_job(worker_id, retry_count=0, max_retries=3, priority=5):
    """Create a RUNNING Job instance assigned to the given worker."""
    now = datetime.now(timezone.utc)
    return Job(
        id=uuid.uuid4(),
        job_type="test_job",
        payload={"data": "test"},
        priority=priority,
        status=JobStatus.RUNNING,
        max_retries=max_retries,
        retry_count=retry_count,
        timeout_seconds=300,
        worker_id=worker_id,
        started_at=now,
        created_at=now,
        updated_at=now,
    )


# ---------------------------------------------------------------------------
# Property Tests
# ---------------------------------------------------------------------------


class TestFailureRecoveryCompleteness:
    """Property 18: Failure Recovery Completeness.

    **Validates: Requirements 8.1, 8.2, 8.3, 8.4**
    """

    @given(job_configs=jobs_list_strategy)
    @hypothesis_settings(max_examples=100, deadline=10000)
    @pytest.mark.asyncio
    async def test_all_running_jobs_are_recovered(self, job_configs):
        """Property 18a: For ANY set of RUNNING jobs assigned to a dead worker,
        ALL jobs are recovered (recovered count == number of RUNNING jobs).

        **Validates: Requirements 8.1, 8.2**
        """
        # Setup fresh database
        engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        session_factory = async_sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )
        fake_redis = FakeRedis()
        fake_pq = FakePriorityQueue()
        detector = FailureDetector(
            session_factory=session_factory,
            redis_client=fake_redis,
            priority_queue=fake_pq,
        )

        # Create worker and jobs
        worker = make_worker()
        jobs = [
            make_job(
                worker_id=worker.id,
                retry_count=cfg["retry_count"],
                max_retries=cfg["max_retries"],
                priority=cfg["priority"],
            )
            for cfg in job_configs
        ]

        async with session_factory() as session:
            async with session.begin():
                session.add(worker)
                session.add_all(jobs)

        # Execute recovery
        recovered = await detector.recover_abandoned_jobs(worker.id)

        # Property: ALL RUNNING jobs must be recovered
        assert recovered == len(jobs), (
            f"Expected {len(jobs)} recovered jobs, got {recovered}"
        )

        await engine.dispose()

    @given(job_configs=jobs_list_strategy)
    @hypothesis_settings(max_examples=100, deadline=10000)
    @pytest.mark.asyncio
    async def test_no_running_jobs_remain_after_recovery(self, job_configs):
        """Property 18b: After recovery, NO job remains in RUNNING status
        assigned to the dead worker.

        **Validates: Requirements 8.1, 8.2**
        """
        engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        session_factory = async_sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )
        fake_redis = FakeRedis()
        fake_pq = FakePriorityQueue()
        detector = FailureDetector(
            session_factory=session_factory,
            redis_client=fake_redis,
            priority_queue=fake_pq,
        )

        worker = make_worker()
        jobs = [
            make_job(
                worker_id=worker.id,
                retry_count=cfg["retry_count"],
                max_retries=cfg["max_retries"],
                priority=cfg["priority"],
            )
            for cfg in job_configs
        ]

        async with session_factory() as session:
            async with session.begin():
                session.add(worker)
                session.add_all(jobs)

        await detector.recover_abandoned_jobs(worker.id)

        # Property: NO job should remain RUNNING for this worker
        async with session_factory() as session:
            stmt = (
                select(Job)
                .where(Job.worker_id == worker.id)
                .where(Job.status == JobStatus.RUNNING)
            )
            result = await session.execute(stmt)
            still_running = result.scalars().all()

        assert len(still_running) == 0, (
            f"Found {len(still_running)} jobs still RUNNING after recovery"
        )

        await engine.dispose()

    @given(job_configs=jobs_list_strategy)
    @hypothesis_settings(max_examples=100, deadline=10000)
    @pytest.mark.asyncio
    async def test_jobs_with_retries_remaining_are_requeued(self, job_configs):
        """Property 18c: Jobs with retry_count < max_retries are ALWAYS
        re-queued (status=QUEUED).

        **Validates: Requirements 8.3**
        """
        engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        session_factory = async_sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )
        fake_redis = FakeRedis()
        fake_pq = FakePriorityQueue()
        detector = FailureDetector(
            session_factory=session_factory,
            redis_client=fake_redis,
            priority_queue=fake_pq,
        )

        worker = make_worker()
        jobs = [
            make_job(
                worker_id=worker.id,
                retry_count=cfg["retry_count"],
                max_retries=cfg["max_retries"],
                priority=cfg["priority"],
            )
            for cfg in job_configs
        ]

        # Track which jobs should be re-queued
        should_requeue_ids = {
            jobs[i].id
            for i, cfg in enumerate(job_configs)
            if cfg["retry_count"] < cfg["max_retries"]
        }

        async with session_factory() as session:
            async with session.begin():
                session.add(worker)
                session.add_all(jobs)

        await detector.recover_abandoned_jobs(worker.id)

        # Property: Every job with retry_count < max_retries must be QUEUED
        if should_requeue_ids:
            async with session_factory() as session:
                stmt = select(Job).where(Job.id.in_(should_requeue_ids))
                result = await session.execute(stmt)
                requeued_jobs = result.scalars().all()

            for job in requeued_jobs:
                assert job.status == JobStatus.QUEUED, (
                    f"Job {job.id} with retry_count < max_retries "
                    f"should be QUEUED but is {job.status}"
                )
                assert job.worker_id is None, (
                    f"Job {job.id} should have worker_id cleared after re-queue"
                )

        await engine.dispose()

    @given(job_configs=jobs_list_strategy)
    @hypothesis_settings(max_examples=100, deadline=10000)
    @pytest.mark.asyncio
    async def test_jobs_with_exhausted_retries_go_to_dlq(self, job_configs):
        """Property 18d: Jobs with retry_count >= max_retries are ALWAYS
        moved to DLQ (status=DEAD_LETTER).

        **Validates: Requirements 8.4**
        """
        engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        session_factory = async_sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )
        fake_redis = FakeRedis()
        fake_pq = FakePriorityQueue()
        detector = FailureDetector(
            session_factory=session_factory,
            redis_client=fake_redis,
            priority_queue=fake_pq,
        )

        worker = make_worker()
        jobs = [
            make_job(
                worker_id=worker.id,
                retry_count=cfg["retry_count"],
                max_retries=cfg["max_retries"],
                priority=cfg["priority"],
            )
            for cfg in job_configs
        ]

        # Track which jobs should go to DLQ
        should_dlq_ids = {
            jobs[i].id
            for i, cfg in enumerate(job_configs)
            if cfg["retry_count"] >= cfg["max_retries"]
        }

        async with session_factory() as session:
            async with session.begin():
                session.add(worker)
                session.add_all(jobs)

        await detector.recover_abandoned_jobs(worker.id)

        # Property: Every job with retry_count >= max_retries must be DEAD_LETTER
        if should_dlq_ids:
            async with session_factory() as session:
                stmt = select(Job).where(Job.id.in_(should_dlq_ids))
                result = await session.execute(stmt)
                dlq_jobs = result.scalars().all()

            for job in dlq_jobs:
                assert job.status == JobStatus.DEAD_LETTER, (
                    f"Job {job.id} with retry_count >= max_retries "
                    f"should be DEAD_LETTER but is {job.status}"
                )
                assert job.worker_id is None, (
                    f"Job {job.id} should have worker_id cleared after DLQ"
                )

        await engine.dispose()
