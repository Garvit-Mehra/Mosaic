"""Property-based tests for Redis reconstructability and consistency reconciliation.

**Validates: Requirements 13.4, 15.4**

Uses Hypothesis to generate arbitrary sets of jobs and verify:
- Property 19: Redis state is fully reconstructable from PostgreSQL data alone
- Property 20: Reconciliation produces consistent state (Redis matches PG)
"""

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from src.core.priority_queue import QUEUE_KEY, calculate_queue_score
from src.core.reconstruction import (
    DLQ_KEY,
    SCHEDULED_QUEUE_KEY,
    StateReconstructor,
)
from src.models.enums import JobStatus


# --- Fake objects for testing ---


class FakeJob:
    """Fake job object mimicking the SQLAlchemy Job model for property tests."""

    def __init__(
        self,
        job_id=None,
        priority=5,
        status=JobStatus.QUEUED,
        execute_at=None,
        max_retries=3,
        retry_count=0,
        worker_id=None,
        created_at=None,
    ):
        self.id = job_id or uuid.uuid4()
        self.job_type = "test_handler"
        self.priority = priority
        self.status = status
        self.execute_at = execute_at
        self.max_retries = max_retries
        self.retry_count = retry_count
        self.worker_id = worker_id
        self.created_at = created_at or datetime(2024, 1, 1, 12, 0, 0)
        self.updated_at = datetime(2024, 1, 1, 12, 0, 0)
        self.started_at = None
        self.error = None


# --- Mock helpers ---


class MockTransactionContext:
    """Mock async transaction context manager."""

    def __init__(self, session):
        self._session = session

    async def __aenter__(self):
        return self._session

    async def __aexit__(self, *args):
        pass


def create_mock_session_factory(queued_jobs, scheduled_jobs, running_jobs):
    """Create a mock session factory returning jobs in sequence for each execute call.

    The StateReconstructor calls execute() three times:
    1. SELECT QUEUED jobs
    2. SELECT SCHEDULED jobs
    3. SELECT RUNNING jobs (inside a transaction)
    """

    class FakeScalarsResult:
        def __init__(self, items):
            self._items = items

        def all(self):
            return self._items

    class FakeExecuteResult:
        def __init__(self, items):
            self._items = items

        def scalars(self):
            return FakeScalarsResult(self._items)

    call_index = [0]
    results_sequence = [queued_jobs, scheduled_jobs, running_jobs]

    class MockSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def execute(self, stmt):
            idx = call_index[0]
            call_index[0] += 1
            if idx < len(results_sequence):
                return FakeExecuteResult(results_sequence[idx])
            return FakeExecuteResult([])

        def begin(self):
            return MockTransactionContext(self)

    session = MockSession()

    class AsyncContextSession:
        async def __aenter__(self):
            return session

        async def __aexit__(self, *args):
            pass

    factory = MagicMock()
    factory.return_value = AsyncContextSession()
    return factory


class TrackingRedis:
    """A fake Redis client that tracks ZADD calls to verify reconstruction correctness."""

    def __init__(self):
        self.deleted_keys = []
        self.sorted_sets = {}  # key -> {member: score}

    async def delete(self, key):
        self.deleted_keys.append(key)
        self.sorted_sets.pop(key, None)
        return 1

    async def zadd(self, key, members):
        if key not in self.sorted_sets:
            self.sorted_sets[key] = {}
        self.sorted_sets[key].update(members)
        return len(members)

    async def zcard(self, key):
        return len(self.sorted_sets.get(key, {}))


# --- Hypothesis strategies ---


job_priority_strategy = st.integers(min_value=0, max_value=100)
retry_count_strategy = st.integers(min_value=0, max_value=10)
max_retries_strategy = st.integers(min_value=1, max_value=10)


def queued_job_strategy():
    """Generate a QUEUED job with random priority and created_at."""
    return st.builds(
        FakeJob,
        job_id=st.builds(uuid.uuid4),
        priority=job_priority_strategy,
        status=st.just(JobStatus.QUEUED),
        created_at=st.datetimes(
            min_value=datetime(2024, 1, 1),
            max_value=datetime(2024, 12, 31),
        ),
    )


def scheduled_job_strategy():
    """Generate a SCHEDULED job with a future execute_at time."""
    # Generate execute_at times far in the future to ensure they pass the > now check
    return st.builds(
        FakeJob,
        job_id=st.builds(uuid.uuid4),
        priority=job_priority_strategy,
        status=st.just(JobStatus.SCHEDULED),
        execute_at=st.datetimes(
            min_value=datetime(2030, 1, 1),
            max_value=datetime(2035, 12, 31),
        ),
    )


def running_job_strategy():
    """Generate a RUNNING job with random retry_count and max_retries."""
    return st.builds(
        FakeJob,
        job_id=st.builds(uuid.uuid4),
        priority=job_priority_strategy,
        status=st.just(JobStatus.RUNNING),
        retry_count=retry_count_strategy,
        max_retries=max_retries_strategy,
        worker_id=st.builds(uuid.uuid4),
    )


# --- Property Tests ---


class TestRedisReconstructability:
    """Property 19: Redis Reconstructability.

    For any set of jobs persisted in PostgreSQL, the priority queue state in Redis
    SHALL be fully reconstructable from PostgreSQL data alone after a complete Redis failure.

    **Validates: Requirements 13.4**
    """

    @given(queued_jobs=st.lists(queued_job_strategy(), min_size=0, max_size=20))
    @settings(max_examples=100)
    async def test_priority_queue_contains_exactly_queued_job_ids_with_correct_scores(
        self, queued_jobs
    ):
        """Property 19a: For ANY set of QUEUED jobs in PG, after reconstruction,
        the priority queue contains exactly those job IDs with correct scores.

        **Validates: Requirements 13.4**
        """
        tracking_redis = TrackingRedis()
        mock_pq = AsyncMock()
        factory = create_mock_session_factory(queued_jobs, [], [])

        reconstructor = StateReconstructor(factory, tracking_redis, mock_pq)
        result = await reconstructor.reconstruct()

        # Verify count matches
        assert result["priority_queue_rebuilt"] == len(queued_jobs)

        if queued_jobs:
            # Priority queue should contain exactly the QUEUED job IDs
            pq_members = tracking_redis.sorted_sets.get(QUEUE_KEY, {})
            actual_ids = set(pq_members.keys())
            expected_ids = {str(job.id) for job in queued_jobs}
            assert actual_ids == expected_ids

            # Verify each job has the correct score
            for job in queued_jobs:
                expected_score = calculate_queue_score(job.priority, job.created_at)
                actual_score = pq_members[str(job.id)]
                assert actual_score == expected_score, (
                    f"Job {job.id} has score {actual_score}, expected {expected_score}"
                )
        else:
            # No jobs -> no ZADD should have been called for priority queue
            assert QUEUE_KEY not in tracking_redis.sorted_sets

    @given(scheduled_jobs=st.lists(scheduled_job_strategy(), min_size=0, max_size=20))
    @settings(max_examples=100)
    async def test_scheduled_set_contains_exactly_scheduled_job_ids_with_correct_scores(
        self, scheduled_jobs
    ):
        """Property 19b: For ANY set of SCHEDULED jobs with future execute_at,
        after reconstruction, the scheduled set contains exactly those job IDs
        with correct execute_at_ms scores.

        **Validates: Requirements 13.4**
        """
        tracking_redis = TrackingRedis()
        mock_pq = AsyncMock()
        factory = create_mock_session_factory([], scheduled_jobs, [])

        reconstructor = StateReconstructor(factory, tracking_redis, mock_pq)
        result = await reconstructor.reconstruct()

        # Verify count matches
        assert result["scheduled_rebuilt"] == len(scheduled_jobs)

        if scheduled_jobs:
            # Scheduled set should contain exactly the SCHEDULED job IDs
            sched_members = tracking_redis.sorted_sets.get(SCHEDULED_QUEUE_KEY, {})
            actual_ids = set(sched_members.keys())
            expected_ids = {str(job.id) for job in scheduled_jobs}
            assert actual_ids == expected_ids

            # Verify each job has score = execute_at timestamp in milliseconds
            for job in scheduled_jobs:
                execute_at = job.execute_at
                if execute_at.tzinfo is None:
                    execute_at = execute_at.replace(tzinfo=timezone.utc)
                expected_score = int(execute_at.timestamp() * 1000)
                actual_score = sched_members[str(job.id)]
                assert actual_score == expected_score, (
                    f"Job {job.id} has score {actual_score}, "
                    f"expected {expected_score} (execute_at={job.execute_at})"
                )
        else:
            # No jobs -> no ZADD should have been called for scheduled set
            assert SCHEDULED_QUEUE_KEY not in tracking_redis.sorted_sets

    @given(running_jobs=st.lists(running_job_strategy(), min_size=0, max_size=20))
    @settings(max_examples=100)
    async def test_running_jobs_are_recovered_correctly_based_on_retry_status(
        self, running_jobs
    ):
        """Property 19c: For ANY set of RUNNING jobs, after reconstruction,
        all are recovered (QUEUED or DEAD_LETTER based on retry status)
        and correctly placed in Redis.

        **Validates: Requirements 13.4**
        """
        # Categorize BEFORE reconstruction (since reconstruct mutates job objects)
        expected_requeued_ids = set()
        expected_dlq_ids = set()
        for job in running_jobs:
            new_retry_count = job.retry_count + 1
            if new_retry_count <= job.max_retries:
                expected_requeued_ids.add(str(job.id))
            else:
                expected_dlq_ids.add(str(job.id))

        tracking_redis = TrackingRedis()
        mock_pq = AsyncMock()
        factory = create_mock_session_factory([], [], running_jobs)

        reconstructor = StateReconstructor(factory, tracking_redis, mock_pq)
        result = await reconstructor.reconstruct()

        # Total recovered should match input count
        assert result["running_recovered"] == len(running_jobs)

        # Verify re-queued jobs are in priority queue
        pq_members = tracking_redis.sorted_sets.get(QUEUE_KEY, {})
        requeued_ids = set(pq_members.keys())
        assert requeued_ids == expected_requeued_ids, (
            f"Expected re-queued: {expected_requeued_ids}, got: {requeued_ids}"
        )

        # Verify DLQ jobs are in dead letter queue
        dlq_members = tracking_redis.sorted_sets.get(DLQ_KEY, {})
        dlq_ids = set(dlq_members.keys())
        assert dlq_ids == expected_dlq_ids, (
            f"Expected DLQ: {expected_dlq_ids}, got: {dlq_ids}"
        )

        # Verify job statuses were updated correctly after reconstruction
        for job in running_jobs:
            if str(job.id) in expected_requeued_ids:
                assert job.status == JobStatus.QUEUED
                assert job.worker_id is None
                assert job.started_at is None
            else:
                assert job.status == JobStatus.DEAD_LETTER
                assert job.worker_id is None


class TestConsistencyReconciliation:
    """Property 20: Consistency Reconciliation.

    For any state divergence between PostgreSQL and Redis, the reconciliation
    process SHALL produce a consistent state where Redis accurately reflects
    the job states stored in PostgreSQL.

    **Validates: Requirements 15.4**
    """

    @given(
        queued_jobs=st.lists(queued_job_strategy(), min_size=0, max_size=10),
        scheduled_jobs=st.lists(scheduled_job_strategy(), min_size=0, max_size=10),
        running_jobs=st.lists(running_job_strategy(), min_size=0, max_size=10),
    )
    @settings(max_examples=100)
    async def test_reconciliation_produces_consistent_redis_state(
        self, queued_jobs, scheduled_jobs, running_jobs
    ):
        """Property 20: After reconstruct(), the number of items in Redis queues
        equals the expected count from PG data (consistency check).

        The reconciliation clears Redis entirely and rebuilds from PG,
        so after reconciliation Redis MUST contain exactly:
        - priority queue: all QUEUED jobs + re-queued RUNNING jobs
        - scheduled set: all SCHEDULED jobs
        - DLQ: RUNNING jobs that exceeded max_retries

        **Validates: Requirements 15.4**
        """
        # Calculate expected counts BEFORE reconstruction (since it mutates jobs)
        expected_requeued_running = sum(
            1 for job in running_jobs
            if (job.retry_count + 1) <= job.max_retries
        )
        expected_dlq_running = sum(
            1 for job in running_jobs
            if (job.retry_count + 1) > job.max_retries
        )

        tracking_redis = TrackingRedis()
        mock_pq = AsyncMock()
        factory = create_mock_session_factory(queued_jobs, scheduled_jobs, running_jobs)

        reconstructor = StateReconstructor(factory, tracking_redis, mock_pq)
        result = await reconstructor.reconcile()

        # Consistency check: Redis queue depths match expected
        expected_priority_count = len(queued_jobs) + expected_requeued_running
        expected_scheduled_count = len(scheduled_jobs)
        expected_dlq_count = expected_dlq_running

        actual_priority_count = len(tracking_redis.sorted_sets.get(QUEUE_KEY, {}))
        actual_scheduled_count = len(
            tracking_redis.sorted_sets.get(SCHEDULED_QUEUE_KEY, {})
        )
        actual_dlq_count = len(tracking_redis.sorted_sets.get(DLQ_KEY, {}))

        assert actual_priority_count == expected_priority_count, (
            f"Priority queue: expected {expected_priority_count}, "
            f"got {actual_priority_count} "
            f"(queued={len(queued_jobs)}, requeued_running={expected_requeued_running})"
        )
        assert actual_scheduled_count == expected_scheduled_count, (
            f"Scheduled set: expected {expected_scheduled_count}, "
            f"got {actual_scheduled_count}"
        )
        assert actual_dlq_count == expected_dlq_count, (
            f"DLQ: expected {expected_dlq_count}, got {actual_dlq_count}"
        )

        # Verify reported counts match actual Redis state
        assert result["priority_queue_rebuilt"] == len(queued_jobs)
        assert result["scheduled_rebuilt"] == len(scheduled_jobs)
        assert result["running_recovered"] == len(running_jobs)

        # Verify reconstruction signals ready state
        assert reconstructor.is_ready

        # Verify all old keys were cleared first (clean slate)
        assert QUEUE_KEY in tracking_redis.deleted_keys
        assert SCHEDULED_QUEUE_KEY in tracking_redis.deleted_keys
        assert DLQ_KEY in tracking_redis.deleted_keys
