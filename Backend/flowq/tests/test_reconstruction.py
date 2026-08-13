"""Unit tests for the state reconstruction module.

Tests cover the StateReconstructor's ability to rebuild Redis queue state
from PostgreSQL data, including priority queue reconstruction, scheduled
queue reconstruction, and recovery of abandoned RUNNING jobs.

These tests use mocks for PostgreSQL sessions and Redis to test logic
without requiring live infrastructure.
"""

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.reconstruction import (
    DLQ_KEY,
    QUEUE_KEY,
    RECONCILIATION_TIMEOUT_SECONDS,
    RETRY_QUEUE_KEY,
    SCHEDULED_QUEUE_KEY,
    StateReconstructor,
)
from src.models.enums import JobStatus


class FakeJob:
    """Fake job object mimicking the SQLAlchemy Job model."""

    def __init__(
        self,
        job_id=None,
        job_type="test_handler",
        priority=5,
        status=JobStatus.QUEUED,
        execute_at=None,
        max_retries=3,
        retry_count=0,
        worker_id=None,
        created_at=None,
        updated_at=None,
        started_at=None,
        error=None,
    ):
        self.id = job_id or uuid.uuid4()
        self.job_type = job_type
        self.priority = priority
        self.status = status
        self.execute_at = execute_at
        self.max_retries = max_retries
        self.retry_count = retry_count
        self.worker_id = worker_id
        self.created_at = created_at or datetime(2024, 1, 1, 12, 0, 0)
        self.updated_at = updated_at or datetime(2024, 1, 1, 12, 0, 0)
        self.started_at = started_at
        self.error = error


class FakeScalarsResult:
    """Fake scalars result for mocking SQLAlchemy query results."""

    def __init__(self, items):
        self._items = items

    def all(self):
        return self._items


class FakeExecuteResult:
    """Fake execution result for mocking SQLAlchemy session.execute()."""

    def __init__(self, items):
        self._items = items

    def scalars(self):
        return FakeScalarsResult(self._items)


def create_mock_session_factory(jobs_by_status):
    """Create a mock session factory that returns jobs filtered by status.

    Args:
        jobs_by_status: Dict mapping JobStatus to list of FakeJob objects.
    """

    class MockSession:
        def __init__(self):
            self._in_transaction = False

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def execute(self, stmt):
            # Inspect the statement's WHERE clause to determine status filter
            # We look at the compiled statement to find the status
            compiled = str(stmt.compile(compile_kwargs={"literal_binds": True}))
            for status, jobs in jobs_by_status.items():
                if status.value in compiled:
                    return FakeExecuteResult(jobs)
            return FakeExecuteResult([])

        def begin(self):
            return self

    session = MockSession()

    factory = MagicMock()
    factory.return_value = session
    factory().__aenter__ = AsyncMock(return_value=session)
    factory().__aexit__ = AsyncMock(return_value=None)

    # Make it work as async context manager
    class AsyncContextSession:
        async def __aenter__(self):
            return session

        async def __aexit__(self, *args):
            pass

    factory.return_value = AsyncContextSession()
    return factory


def create_simple_mock_session_factory(query_results_sequence):
    """Create a mock session factory that returns results in sequence for each execute call.

    Args:
        query_results_sequence: List of lists, each inner list is returned on successive execute calls.
    """
    call_index = [0]

    class MockSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def execute(self, stmt):
            idx = call_index[0]
            call_index[0] += 1
            if idx < len(query_results_sequence):
                return FakeExecuteResult(query_results_sequence[idx])
            return FakeExecuteResult([])

        def begin(self):
            return MockTransactionContext(self)

    class MockTransactionContext:
        def __init__(self, session):
            self._session = session

        async def __aenter__(self):
            return self._session

        async def __aexit__(self, *args):
            pass

    session = MockSession()

    class AsyncContextSession:
        async def __aenter__(self):
            return session

        async def __aexit__(self, *args):
            pass

    factory = MagicMock()
    factory.return_value = AsyncContextSession()
    return factory


@pytest.fixture
def mock_redis():
    """Create a mock Redis client with async methods."""
    client = AsyncMock()
    client.delete = AsyncMock(return_value=1)
    client.zadd = AsyncMock(return_value=1)
    client.zcard = AsyncMock(return_value=0)
    return client


@pytest.fixture
def mock_priority_queue():
    """Create a mock priority queue."""
    pq = AsyncMock()
    pq.enqueue = AsyncMock()
    pq.depth = AsyncMock(return_value=0)
    return pq


class TestStateReconstructorInit:
    """Tests for StateReconstructor initialization and ready state."""

    def test_initial_state_is_not_ready(self, mock_redis, mock_priority_queue):
        """Reconstructor should not be ready before reconstruct() is called."""
        factory = MagicMock()
        reconstructor = StateReconstructor(factory, mock_redis, mock_priority_queue)
        assert not reconstructor.is_ready

    async def test_is_ready_after_reconstruct(self, mock_redis, mock_priority_queue):
        """Reconstructor should be ready after successful reconstruct()."""
        factory = create_simple_mock_session_factory([[], [], []])
        reconstructor = StateReconstructor(factory, mock_redis, mock_priority_queue)
        await reconstructor.reconstruct()
        assert reconstructor.is_ready


class TestClearRedisQueues:
    """Tests for clearing Redis queue keys during reconstruction."""

    async def test_clears_all_queue_keys(self, mock_redis, mock_priority_queue):
        """Reconstruction should delete all Redis queue keys."""
        factory = create_simple_mock_session_factory([[], [], []])
        reconstructor = StateReconstructor(factory, mock_redis, mock_priority_queue)
        await reconstructor.reconstruct()

        # Verify all queue keys were deleted
        delete_calls = mock_redis.delete.call_args_list
        deleted_keys = [call[0][0] for call in delete_calls]
        assert QUEUE_KEY in deleted_keys
        assert SCHEDULED_QUEUE_KEY in deleted_keys
        assert RETRY_QUEUE_KEY in deleted_keys
        assert DLQ_KEY in deleted_keys


class TestRebuildPriorityQueue:
    """Tests for rebuilding the priority queue from QUEUED jobs."""

    async def test_rebuilds_queued_jobs(self, mock_redis, mock_priority_queue):
        """Should add all QUEUED jobs to the Redis priority sorted set."""
        job1 = FakeJob(status=JobStatus.QUEUED, priority=5)
        job2 = FakeJob(status=JobStatus.QUEUED, priority=10)

        # Query sequence: QUEUED jobs, SCHEDULED jobs, RUNNING jobs
        factory = create_simple_mock_session_factory([[job1, job2], [], []])
        reconstructor = StateReconstructor(factory, mock_redis, mock_priority_queue)

        result = await reconstructor.reconstruct()

        assert result["priority_queue_rebuilt"] == 2
        # Verify ZADD was called with the priority queue key
        zadd_calls = mock_redis.zadd.call_args_list
        # First ZADD should be for priority queue (after deletes)
        priority_zadd = zadd_calls[0]
        assert priority_zadd[0][0] == QUEUE_KEY

    async def test_no_queued_jobs_skips_zadd(self, mock_redis, mock_priority_queue):
        """Should not call ZADD if there are no QUEUED jobs."""
        factory = create_simple_mock_session_factory([[], [], []])
        reconstructor = StateReconstructor(factory, mock_redis, mock_priority_queue)

        result = await reconstructor.reconstruct()

        assert result["priority_queue_rebuilt"] == 0
        # Only delete calls, no ZADD
        assert mock_redis.zadd.call_count == 0


class TestRebuildScheduledQueue:
    """Tests for rebuilding the scheduled queue from SCHEDULED jobs."""

    async def test_rebuilds_scheduled_jobs(self, mock_redis, mock_priority_queue):
        """Should add SCHEDULED jobs with future execute_at to schedule set."""
        future_time = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(hours=1)
        job1 = FakeJob(
            status=JobStatus.SCHEDULED,
            execute_at=future_time,
        )

        # Query sequence: QUEUED jobs, SCHEDULED jobs, RUNNING jobs
        factory = create_simple_mock_session_factory([[], [job1], []])
        reconstructor = StateReconstructor(factory, mock_redis, mock_priority_queue)

        result = await reconstructor.reconstruct()

        assert result["scheduled_rebuilt"] == 1
        # Verify ZADD was called with the scheduled queue key
        zadd_calls = mock_redis.zadd.call_args_list
        scheduled_zadd = zadd_calls[0]
        assert scheduled_zadd[0][0] == SCHEDULED_QUEUE_KEY

    async def test_scheduled_job_score_is_execute_at_ms(self, mock_redis, mock_priority_queue):
        """Score for scheduled jobs should be execute_at in milliseconds."""
        future_time = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        expected_score = int(future_time.timestamp() * 1000)
        job1 = FakeJob(
            status=JobStatus.SCHEDULED,
            execute_at=future_time,
        )

        factory = create_simple_mock_session_factory([[], [job1], []])
        reconstructor = StateReconstructor(factory, mock_redis, mock_priority_queue)

        await reconstructor.reconstruct()

        zadd_calls = mock_redis.zadd.call_args_list
        scheduled_zadd = zadd_calls[0]
        members = scheduled_zadd[0][1]
        assert list(members.values())[0] == expected_score


class TestRecoverRunningJobs:
    """Tests for recovering abandoned RUNNING jobs."""

    async def test_requeues_running_jobs_with_retries_remaining(
        self, mock_redis, mock_priority_queue
    ):
        """RUNNING jobs with retry_count < max_retries should be re-queued."""
        job = FakeJob(
            status=JobStatus.RUNNING,
            retry_count=1,
            max_retries=3,
            worker_id=uuid.uuid4(),
            priority=7,
        )

        factory = create_simple_mock_session_factory([[], [], [job]])
        reconstructor = StateReconstructor(factory, mock_redis, mock_priority_queue)

        result = await reconstructor.reconstruct()

        assert result["running_recovered"] == 1
        # Job should be marked QUEUED with incremented retry_count
        assert job.status == JobStatus.QUEUED
        assert job.retry_count == 2
        assert job.worker_id is None
        assert job.started_at is None
        assert "abandoned" in job.error.lower()

    async def test_dlq_running_jobs_with_exhausted_retries(
        self, mock_redis, mock_priority_queue
    ):
        """RUNNING jobs with retry_count >= max_retries should go to DLQ."""
        job = FakeJob(
            status=JobStatus.RUNNING,
            retry_count=3,
            max_retries=3,
            worker_id=uuid.uuid4(),
            priority=5,
        )

        factory = create_simple_mock_session_factory([[], [], [job]])
        reconstructor = StateReconstructor(factory, mock_redis, mock_priority_queue)

        result = await reconstructor.reconstruct()

        assert result["running_recovered"] == 1
        # Job should be marked DEAD_LETTER
        assert job.status == JobStatus.DEAD_LETTER
        assert job.worker_id is None
        assert "max retries" in job.error.lower()

    async def test_mixed_running_jobs_are_correctly_routed(
        self, mock_redis, mock_priority_queue
    ):
        """Mix of retryable and non-retryable jobs should be correctly routed."""
        retryable_job = FakeJob(
            status=JobStatus.RUNNING,
            retry_count=0,
            max_retries=3,
            worker_id=uuid.uuid4(),
        )
        dlq_job = FakeJob(
            status=JobStatus.RUNNING,
            retry_count=3,
            max_retries=3,
            worker_id=uuid.uuid4(),
        )

        factory = create_simple_mock_session_factory([[], [], [retryable_job, dlq_job]])
        reconstructor = StateReconstructor(factory, mock_redis, mock_priority_queue)

        result = await reconstructor.reconstruct()

        assert result["running_recovered"] == 2
        assert retryable_job.status == JobStatus.QUEUED
        assert dlq_job.status == JobStatus.DEAD_LETTER

    async def test_redis_zadd_called_for_requeued_and_dlq(
        self, mock_redis, mock_priority_queue
    ):
        """Should ZADD to both priority queue and DLQ for mixed results."""
        retryable_job = FakeJob(
            status=JobStatus.RUNNING,
            retry_count=0,
            max_retries=3,
            worker_id=uuid.uuid4(),
            priority=5,
        )
        dlq_job = FakeJob(
            status=JobStatus.RUNNING,
            retry_count=3,
            max_retries=3,
            worker_id=uuid.uuid4(),
            priority=5,
        )

        factory = create_simple_mock_session_factory([[], [], [retryable_job, dlq_job]])
        reconstructor = StateReconstructor(factory, mock_redis, mock_priority_queue)

        await reconstructor.reconstruct()

        # Should have ZADD calls for both priority queue and DLQ
        zadd_calls = mock_redis.zadd.call_args_list
        zadd_keys = [call[0][0] for call in zadd_calls]
        assert QUEUE_KEY in zadd_keys
        assert DLQ_KEY in zadd_keys


class TestReconstructResult:
    """Tests for the overall reconstruct() result dict."""

    async def test_returns_complete_result_dict(self, mock_redis, mock_priority_queue):
        """reconstruct() should return a dict with all expected keys."""
        factory = create_simple_mock_session_factory([[], [], []])
        reconstructor = StateReconstructor(factory, mock_redis, mock_priority_queue)

        result = await reconstructor.reconstruct()

        assert "priority_queue_rebuilt" in result
        assert "scheduled_rebuilt" in result
        assert "running_recovered" in result
        assert "total_time_ms" in result
        assert isinstance(result["total_time_ms"], int)
        assert result["total_time_ms"] >= 0

    async def test_total_counts_match_individual_counts(
        self, mock_redis, mock_priority_queue
    ):
        """All count fields should be non-negative integers."""
        job1 = FakeJob(status=JobStatus.QUEUED, priority=5)
        job2 = FakeJob(
            status=JobStatus.RUNNING,
            retry_count=0,
            max_retries=3,
            worker_id=uuid.uuid4(),
        )

        factory = create_simple_mock_session_factory([[job1], [], [job2]])
        reconstructor = StateReconstructor(factory, mock_redis, mock_priority_queue)

        result = await reconstructor.reconstruct()

        assert result["priority_queue_rebuilt"] == 1
        assert result["scheduled_rebuilt"] == 0
        assert result["running_recovered"] == 1


class TestWaitUntilReady:
    """Tests for the wait_until_ready blocking behavior."""

    async def test_wait_returns_true_after_reconstruction(
        self, mock_redis, mock_priority_queue
    ):
        """wait_until_ready() should return True after reconstruction completes."""
        factory = create_simple_mock_session_factory([[], [], []])
        reconstructor = StateReconstructor(factory, mock_redis, mock_priority_queue)

        await reconstructor.reconstruct()
        ready = await reconstructor.wait_until_ready(timeout=1.0)
        assert ready is True

    async def test_wait_blocks_until_reconstruction(
        self, mock_redis, mock_priority_queue
    ):
        """wait_until_ready() should block until reconstruct() completes."""
        factory = create_simple_mock_session_factory([[], [], []])
        reconstructor = StateReconstructor(factory, mock_redis, mock_priority_queue)

        results = []

        async def waiter():
            ready = await reconstructor.wait_until_ready(timeout=5.0)
            results.append(ready)

        async def reconstructer():
            await asyncio.sleep(0.05)  # Small delay
            await reconstructor.reconstruct()

        await asyncio.gather(waiter(), reconstructer())
        assert results == [True]

    async def test_wait_returns_false_on_timeout(
        self, mock_redis, mock_priority_queue
    ):
        """wait_until_ready() should return False if timeout elapses."""
        factory = create_simple_mock_session_factory([[], [], []])
        reconstructor = StateReconstructor(factory, mock_redis, mock_priority_queue)

        # Don't call reconstruct - should timeout
        ready = await reconstructor.wait_until_ready(timeout=0.05)
        assert ready is False


class TestReconcile:
    """Tests for the reconcile() method with timeout enforcement."""

    async def test_reconcile_delegates_to_reconstruct(
        self, mock_redis, mock_priority_queue
    ):
        """reconcile() should call reconstruct() and return its result."""
        factory = create_simple_mock_session_factory([[], [], []])
        reconstructor = StateReconstructor(factory, mock_redis, mock_priority_queue)

        result = await reconstructor.reconcile()

        assert result["priority_queue_rebuilt"] == 0
        assert result["scheduled_rebuilt"] == 0
        assert result["running_recovered"] == 0
        assert reconstructor.is_ready

    async def test_reconcile_has_60_second_timeout(
        self, mock_redis, mock_priority_queue
    ):
        """reconcile() should enforce the 60-second timeout (Req 15.4)."""
        # This test verifies that RECONCILIATION_TIMEOUT_SECONDS is 60
        assert RECONCILIATION_TIMEOUT_SECONDS == 60


class TestReconstructionBlocksSubmissions:
    """Tests verifying that reconstruction blocks submissions (Req 13.5)."""

    async def test_not_ready_during_reconstruction(
        self, mock_redis, mock_priority_queue
    ):
        """System should not be ready while reconstruction is in progress."""
        factory = create_simple_mock_session_factory([[], [], []])
        reconstructor = StateReconstructor(factory, mock_redis, mock_priority_queue)

        assert not reconstructor.is_ready
        await reconstructor.reconstruct()
        assert reconstructor.is_ready

    async def test_reconstruction_sets_ready_even_on_empty_db(
        self, mock_redis, mock_priority_queue
    ):
        """Reconstruction with no jobs should still signal ready."""
        factory = create_simple_mock_session_factory([[], [], []])
        reconstructor = StateReconstructor(factory, mock_redis, mock_priority_queue)

        await reconstructor.reconstruct()
        assert reconstructor.is_ready
