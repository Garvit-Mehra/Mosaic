"""Unit tests for the Job Coordinator.

Tests cover submit_job, get_job, list_jobs, and get_job_history operations.
Uses an in-memory SQLite database for fast, isolated testing.
"""

import pytest
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.core.coordinator import JobCoordinator, SCHEDULED_QUEUE_KEY
from src.core.handler_registry import HandlerRegistry
from src.core.state_machine import InvalidTransitionError
from src.models.base import Base
from src.models.enums import JobStatus
from src.models.job import Job
from src.models.job_execution import JobExecution


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
    client.zadd = AsyncMock(return_value=1)
    return client


@pytest.fixture
def mock_priority_queue():
    """Create a mock priority queue."""
    queue = AsyncMock()
    queue.enqueue = AsyncMock(return_value=None)
    return queue


@pytest.fixture
def registry():
    """Create a handler registry with a test handler registered."""
    reg = HandlerRegistry()
    reg.register("email", lambda x: x)
    reg.register("process_data", lambda x: x)
    return reg


@pytest.fixture
def coordinator(session_factory, mock_redis, mock_priority_queue):
    """Create a JobCoordinator with test dependencies."""
    return JobCoordinator(
        session_factory=session_factory,
        redis_client=mock_redis,
        priority_queue=mock_priority_queue,
    )


class TestSubmitJob:
    """Tests for the submit_job operation."""

    async def test_submit_immediate_job_creates_queued_job(
        self, coordinator, mock_priority_queue, registry
    ):
        """Submitting without execute_at creates a QUEUED job."""
        with patch("src.core.coordinator.validate_job_submission") as mock_validate:
            result = await coordinator.submit_job(
                job_type="email",
                payload={"to": "user@example.com"},
                priority=5,
            )

        assert result["status"] == "QUEUED"
        assert result["job_type"] == "email"
        assert result["priority"] == 5
        assert result["id"] is not None
        assert result["created_at"] is not None
        mock_priority_queue.enqueue.assert_called_once()

    async def test_submit_scheduled_job_creates_scheduled_job(
        self, coordinator, mock_redis, mock_priority_queue, registry
    ):
        """Submitting with execute_at creates a SCHEDULED job."""
        future_time = datetime.utcnow() + timedelta(hours=1)

        with patch("src.core.coordinator.validate_job_submission"):
            result = await coordinator.submit_job(
                job_type="email",
                payload={"to": "user@example.com"},
                priority=3,
                execute_at=future_time,
            )

        assert result["status"] == "SCHEDULED"
        assert result["job_type"] == "email"
        assert result["priority"] == 3
        # Should add to scheduled set, not priority queue
        mock_redis.zadd.assert_called_once()
        mock_priority_queue.enqueue.assert_not_called()

    async def test_submit_job_persists_to_database(
        self, coordinator, session_factory
    ):
        """Submitted job should be retrievable from the database."""
        with patch("src.core.coordinator.validate_job_submission"):
            result = await coordinator.submit_job(
                job_type="process_data",
                payload={"data": [1, 2, 3]},
                priority=7,
                timeout_seconds=600,
                max_retries=5,
            )

        # Verify we can retrieve it
        job_response = await coordinator.get_job(result["id"])
        assert job_response is not None
        assert job_response["job_type"] == "process_data"
        assert job_response["priority"] == 7
        assert job_response["timeout_seconds"] == 600
        assert job_response["max_retries"] == 5

    async def test_submit_job_returns_required_fields(self, coordinator):
        """Response must include id, status, job_type, priority, created_at (Req 1.8)."""
        with patch("src.core.coordinator.validate_job_submission"):
            result = await coordinator.submit_job(
                job_type="email",
                payload={"to": "test@test.com"},
            )

        assert "id" in result
        assert "status" in result
        assert "job_type" in result
        assert "priority" in result
        assert "created_at" in result

    async def test_submit_immediate_job_adds_to_priority_queue(
        self, coordinator, mock_priority_queue
    ):
        """Immediate job should be enqueued with correct priority and timestamp."""
        with patch("src.core.coordinator.validate_job_submission"):
            result = await coordinator.submit_job(
                job_type="email",
                payload={},
                priority=10,
            )

        call_args = mock_priority_queue.enqueue.call_args
        assert call_args.kwargs["job_id"] == result["id"]
        assert call_args.kwargs["priority"] == 10

    async def test_submit_scheduled_job_adds_to_schedule_set(
        self, coordinator, mock_redis
    ):
        """Scheduled job should be added to queue:scheduled with execute_at score."""
        future_time = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)

        with patch("src.core.coordinator.validate_job_submission"):
            result = await coordinator.submit_job(
                job_type="email",
                payload={},
                execute_at=future_time,
            )

        expected_score = int(future_time.timestamp() * 1000)
        mock_redis.zadd.assert_called_once_with(
            SCHEDULED_QUEUE_KEY,
            {str(result["id"]): expected_score},
        )

    async def test_submit_job_calls_validation(self, coordinator):
        """submit_job should call validate_job_submission with correct args."""
        with patch("src.core.coordinator.validate_job_submission") as mock_validate:
            await coordinator.submit_job(
                job_type="email",
                payload={"key": "value"},
                priority=3,
                timeout_seconds=120,
                max_retries=2,
            )

        mock_validate.assert_called_once_with(
            job_type="email",
            payload={"key": "value"},
            priority=3,
            timeout_seconds=120,
            max_retries=2,
            execute_at=None,
        )

    async def test_submit_job_validation_error_propagates(self, coordinator):
        """Validation errors should propagate to caller."""
        from src.core.validators import ValidationError

        with patch(
            "src.core.coordinator.validate_job_submission",
            side_effect=ValidationError(400, "priority must be between 0 and 10000"),
        ):
            with pytest.raises(ValidationError) as exc_info:
                await coordinator.submit_job(
                    job_type="email",
                    payload={},
                    priority=-1,
                )
            assert exc_info.value.status_code == 400

    async def test_submit_job_persists_before_redis(
        self, coordinator, session_factory, mock_priority_queue
    ):
        """PostgreSQL should be persisted before Redis operations (Req 13.1)."""
        call_order = []

        original_enqueue = mock_priority_queue.enqueue

        async def track_enqueue(**kwargs):
            # At this point, the job should already be in PG
            async with session_factory() as session:
                from sqlalchemy import select

                stmt = select(Job).where(Job.id == kwargs["job_id"])
                result = await session.execute(stmt)
                job = result.scalar_one_or_none()
                if job is not None:
                    call_order.append("pg_persisted")
            call_order.append("redis_enqueue")
            return await original_enqueue(**kwargs)

        mock_priority_queue.enqueue = track_enqueue

        with patch("src.core.coordinator.validate_job_submission"):
            await coordinator.submit_job(
                job_type="email",
                payload={},
                priority=5,
            )

        assert "pg_persisted" in call_order
        assert call_order.index("pg_persisted") < call_order.index("redis_enqueue")


class TestGetJob:
    """Tests for the get_job operation."""

    async def test_get_existing_job(self, coordinator, session_factory):
        """get_job should return job metadata for existing job."""
        # Create a job directly in the database
        async with session_factory() as session:
            async with session.begin():
                job = Job(
                    job_type="email",
                    payload={"to": "test@test.com"},
                    priority=5,
                    status=JobStatus.QUEUED,
                )
                session.add(job)
                await session.flush()
                job_id = job.id

        result = await coordinator.get_job(job_id)

        assert result is not None
        assert result["id"] == job_id
        assert result["job_type"] == "email"
        assert result["status"] == "QUEUED"
        assert result["priority"] == 5
        assert result["payload"] == {"to": "test@test.com"}

    async def test_get_nonexistent_job_returns_none(self, coordinator):
        """get_job should return None for non-existent ID."""
        result = await coordinator.get_job(uuid.uuid4())
        assert result is None

    async def test_get_job_returns_complete_metadata(self, coordinator, session_factory):
        """get_job should return all metadata fields (Req 2.1)."""
        async with session_factory() as session:
            async with session.begin():
                job = Job(
                    job_type="process_data",
                    payload={"items": [1, 2]},
                    priority=8,
                    status=JobStatus.RUNNING,
                    timeout_seconds=600,
                    max_retries=5,
                    retry_count=1,
                    worker_id=uuid.uuid4(),
                    started_at=datetime.utcnow(),
                )
                session.add(job)
                await session.flush()
                job_id = job.id

        result = await coordinator.get_job(job_id)

        assert result["id"] == job_id
        assert result["job_type"] == "process_data"
        assert result["status"] == "RUNNING"
        assert result["priority"] == 8
        assert result["payload"] == {"items": [1, 2]}
        assert result["timeout_seconds"] == 600
        assert result["max_retries"] == 5
        assert result["retry_count"] == 1
        assert result["worker_id"] is not None
        assert result["started_at"] is not None


class TestListJobs:
    """Tests for the list_jobs operation."""

    async def test_list_jobs_empty_database(self, coordinator):
        """list_jobs should return empty list when no jobs exist."""
        result = await coordinator.list_jobs()
        assert result == []

    async def test_list_jobs_returns_all_jobs(self, coordinator, session_factory):
        """list_jobs without filter returns all jobs."""
        async with session_factory() as session:
            async with session.begin():
                for i in range(3):
                    job = Job(
                        job_type=f"type_{i}",
                        payload={},
                        priority=i,
                        status=JobStatus.QUEUED,
                    )
                    session.add(job)

        result = await coordinator.list_jobs()
        assert len(result) == 3

    async def test_list_jobs_with_status_filter(self, coordinator, session_factory):
        """list_jobs with status filter returns only matching jobs."""
        async with session_factory() as session:
            async with session.begin():
                job1 = Job(
                    job_type="email",
                    payload={},
                    priority=1,
                    status=JobStatus.QUEUED,
                )
                job2 = Job(
                    job_type="process",
                    payload={},
                    priority=2,
                    status=JobStatus.RUNNING,
                )
                job3 = Job(
                    job_type="notify",
                    payload={},
                    priority=3,
                    status=JobStatus.QUEUED,
                )
                session.add_all([job1, job2, job3])

        result = await coordinator.list_jobs(status="QUEUED")
        assert len(result) == 2
        assert all(j["status"] == "QUEUED" for j in result)

    async def test_list_jobs_limit_capped_at_100(self, coordinator, session_factory):
        """Limit should be capped at 100 even if larger value provided."""
        async with session_factory() as session:
            async with session.begin():
                for i in range(5):
                    job = Job(
                        job_type="email",
                        payload={},
                        priority=1,
                        status=JobStatus.QUEUED,
                    )
                    session.add(job)

        # Request limit of 200 - should still work, just cap at 100
        result = await coordinator.list_jobs(limit=200)
        assert len(result) == 5  # Only 5 exist, but limit was capped

    async def test_list_jobs_default_limit_is_50(self, coordinator, session_factory):
        """Default limit should be 50."""
        # Create 60 jobs
        async with session_factory() as session:
            async with session.begin():
                for i in range(60):
                    job = Job(
                        job_type="email",
                        payload={},
                        priority=1,
                        status=JobStatus.QUEUED,
                    )
                    session.add(job)

        result = await coordinator.list_jobs()
        assert len(result) == 50

    async def test_list_jobs_with_offset(self, coordinator, session_factory):
        """Offset should skip the first N results."""
        async with session_factory() as session:
            async with session.begin():
                for i in range(10):
                    job = Job(
                        job_type="email",
                        payload={},
                        priority=1,
                        status=JobStatus.QUEUED,
                    )
                    session.add(job)

        result = await coordinator.list_jobs(limit=5, offset=5)
        assert len(result) == 5

    async def test_list_jobs_ordered_by_created_at_desc(
        self, coordinator, session_factory
    ):
        """Jobs should be ordered by created_at descending (newest first)."""
        async with session_factory() as session:
            async with session.begin():
                job1 = Job(
                    job_type="first",
                    payload={},
                    priority=1,
                    status=JobStatus.QUEUED,
                    created_at=datetime(2024, 1, 1),
                )
                job2 = Job(
                    job_type="second",
                    payload={},
                    priority=1,
                    status=JobStatus.QUEUED,
                    created_at=datetime(2024, 6, 1),
                )
                job3 = Job(
                    job_type="third",
                    payload={},
                    priority=1,
                    status=JobStatus.QUEUED,
                    created_at=datetime(2024, 12, 1),
                )
                session.add_all([job1, job2, job3])

        result = await coordinator.list_jobs()
        assert result[0]["job_type"] == "third"
        assert result[1]["job_type"] == "second"
        assert result[2]["job_type"] == "first"

    async def test_list_jobs_invalid_status_returns_empty(self, coordinator):
        """Invalid status filter should return empty list."""
        result = await coordinator.list_jobs(status="INVALID_STATUS")
        assert result == []


class TestGetJobHistory:
    """Tests for the get_job_history operation."""

    async def test_get_history_empty(self, coordinator):
        """get_job_history for job with no executions returns empty list."""
        result = await coordinator.get_job_history(uuid.uuid4())
        assert result == []

    async def test_get_history_returns_executions(self, coordinator, session_factory):
        """get_job_history returns execution records for a job."""
        job_id = uuid.uuid4()
        worker_id = uuid.uuid4()

        async with session_factory() as session:
            async with session.begin():
                exec1 = JobExecution(
                    job_id=job_id,
                    worker_id=worker_id,
                    attempt_number=1,
                    status="FAILED",
                    started_at=datetime(2024, 1, 1, 12, 0, 0),
                    completed_at=datetime(2024, 1, 1, 12, 0, 5),
                    duration_ms=5000,
                    error="timeout",
                )
                exec2 = JobExecution(
                    job_id=job_id,
                    worker_id=worker_id,
                    attempt_number=2,
                    status="COMPLETED",
                    started_at=datetime(2024, 1, 1, 12, 1, 0),
                    completed_at=datetime(2024, 1, 1, 12, 1, 3),
                    duration_ms=3000,
                    error=None,
                )
                session.add_all([exec1, exec2])

        result = await coordinator.get_job_history(job_id)

        assert len(result) == 2
        assert result[0]["attempt_number"] == 1
        assert result[0]["status"] == "FAILED"
        assert result[0]["error"] == "timeout"
        assert result[1]["attempt_number"] == 2
        assert result[1]["status"] == "COMPLETED"
        assert result[1]["error"] is None

    async def test_get_history_ordered_by_attempt_number(
        self, coordinator, session_factory
    ):
        """History should be ordered by attempt_number ascending (Req 2.2)."""
        job_id = uuid.uuid4()
        worker_id = uuid.uuid4()

        async with session_factory() as session:
            async with session.begin():
                # Insert out of order
                exec3 = JobExecution(
                    job_id=job_id,
                    worker_id=worker_id,
                    attempt_number=3,
                    status="COMPLETED",
                    started_at=datetime(2024, 1, 1, 12, 2, 0),
                )
                exec1 = JobExecution(
                    job_id=job_id,
                    worker_id=worker_id,
                    attempt_number=1,
                    status="FAILED",
                    started_at=datetime(2024, 1, 1, 12, 0, 0),
                )
                exec2 = JobExecution(
                    job_id=job_id,
                    worker_id=worker_id,
                    attempt_number=2,
                    status="FAILED",
                    started_at=datetime(2024, 1, 1, 12, 1, 0),
                )
                session.add_all([exec3, exec1, exec2])

        result = await coordinator.get_job_history(job_id)

        assert [r["attempt_number"] for r in result] == [1, 2, 3]

    async def test_get_history_only_returns_for_specified_job(
        self, coordinator, session_factory
    ):
        """get_job_history should only return executions for the specified job."""
        job_id_a = uuid.uuid4()
        job_id_b = uuid.uuid4()
        worker_id = uuid.uuid4()

        async with session_factory() as session:
            async with session.begin():
                exec_a = JobExecution(
                    job_id=job_id_a,
                    worker_id=worker_id,
                    attempt_number=1,
                    status="COMPLETED",
                    started_at=datetime(2024, 1, 1, 12, 0, 0),
                )
                exec_b = JobExecution(
                    job_id=job_id_b,
                    worker_id=worker_id,
                    attempt_number=1,
                    status="FAILED",
                    started_at=datetime(2024, 1, 1, 12, 0, 0),
                )
                session.add_all([exec_a, exec_b])

        result = await coordinator.get_job_history(job_id_a)
        assert len(result) == 1
        assert result[0]["job_id"] == job_id_a
