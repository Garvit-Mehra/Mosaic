"""Unit tests for the job failure handler."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.failure_handler import (
    DLQ_KEY,
    RETRY_QUEUE_KEY,
    handle_job_failure,
    _record_execution,
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


class TestHandleJobFailureRetry:
    """Tests for failure handling when retries remain."""

    @pytest.fixture
    def session(self):
        mock = AsyncMock()
        mock.add = MagicMock()
        mock.flush = AsyncMock()
        return mock

    @pytest.fixture
    def redis_client(self):
        mock = AsyncMock()
        mock.zadd = AsyncMock()
        return mock

    @pytest.mark.asyncio
    async def test_increments_retry_count(self, session, redis_client):
        """Req 9.1: Failed job with retries remaining increments retry_count."""
        job = FakeJob(retry_count=0, max_retries=3)

        await handle_job_failure(job, "connection timeout", session, redis_client)

        assert job.retry_count == 1

    @pytest.mark.asyncio
    async def test_transitions_to_queued_for_retry(self, session, redis_client):
        """Req 9.1: Job transitions through FAILED to QUEUED for retry."""
        job = FakeJob(retry_count=0, max_retries=3)

        await handle_job_failure(job, "connection timeout", session, redis_client)

        assert job.status == JobStatus.QUEUED

    @pytest.mark.asyncio
    async def test_adds_to_retry_queue(self, session, redis_client):
        """Req 9.1: Job is added to retry queue with backoff score."""
        job = FakeJob(retry_count=0, max_retries=3)

        await handle_job_failure(job, "connection timeout", session, redis_client)

        redis_client.zadd.assert_called_once()
        call_args = redis_client.zadd.call_args
        assert call_args[0][0] == RETRY_QUEUE_KEY
        # Score should be a future timestamp in milliseconds
        score_mapping = call_args[0][1]
        job_id_str = str(job.id)
        assert job_id_str in score_mapping
        score = score_mapping[job_id_str]
        now_ms = int(datetime.now(timezone.utc).replace(tzinfo=None).timestamp() * 1000)
        assert score > now_ms  # Retry is in the future

    @pytest.mark.asyncio
    async def test_clears_worker_id(self, session, redis_client):
        """Failed jobs clear worker assignment."""
        job = FakeJob(retry_count=0, max_retries=3)
        assert job.worker_id is not None

        await handle_job_failure(job, "error", session, redis_client)

        assert job.worker_id is None

    @pytest.mark.asyncio
    async def test_clears_started_at_for_retry(self, session, redis_client):
        """Retry jobs clear started_at for fresh execution."""
        job = FakeJob(retry_count=0, max_retries=3)

        await handle_job_failure(job, "error", session, redis_client)

        assert job.started_at is None

    @pytest.mark.asyncio
    async def test_stores_error_message(self, session, redis_client):
        """Failed jobs store the error message."""
        job = FakeJob(retry_count=0, max_retries=3)
        error_msg = "Database connection refused"

        await handle_job_failure(job, error_msg, session, redis_client)

        assert job.error == error_msg

    @pytest.mark.asyncio
    async def test_retry_at_boundary(self, session, redis_client):
        """Req 9.5: Job at max_retries - 1 can still retry (last attempt)."""
        job = FakeJob(retry_count=2, max_retries=3)

        await handle_job_failure(job, "error", session, redis_client)

        # retry_count becomes 3, which equals max_retries, so it should still retry
        assert job.retry_count == 3
        assert job.status == JobStatus.QUEUED
        redis_client.zadd.assert_called_once()
        assert redis_client.zadd.call_args[0][0] == RETRY_QUEUE_KEY

    @pytest.mark.asyncio
    async def test_records_execution_history(self, session, redis_client):
        """Each failure records execution history."""
        job = FakeJob(retry_count=1, max_retries=3)

        await handle_job_failure(job, "error", session, redis_client)

        # session.add should be called for both the job and the execution record
        assert session.add.call_count >= 2
        session.flush.assert_awaited_once()


class TestHandleJobFailureDLQ:
    """Tests for failure handling when retries are exhausted."""

    @pytest.fixture
    def session(self):
        mock = AsyncMock()
        mock.add = MagicMock()
        mock.flush = AsyncMock()
        return mock

    @pytest.fixture
    def redis_client(self):
        mock = AsyncMock()
        mock.zadd = AsyncMock()
        return mock

    @pytest.mark.asyncio
    async def test_transitions_to_dead_letter(self, session, redis_client):
        """Req 10.1: Exhausted retries transitions to DEAD_LETTER."""
        job = FakeJob(retry_count=3, max_retries=3)

        await handle_job_failure(job, "final failure", session, redis_client)

        assert job.status == JobStatus.DEAD_LETTER

    @pytest.mark.asyncio
    async def test_adds_to_dlq(self, session, redis_client):
        """Req 10.1: Exhausted retries adds to DLQ sorted set."""
        job = FakeJob(retry_count=3, max_retries=3)

        await handle_job_failure(job, "final failure", session, redis_client)

        redis_client.zadd.assert_called_once()
        call_args = redis_client.zadd.call_args
        assert call_args[0][0] == DLQ_KEY
        score_mapping = call_args[0][1]
        assert str(job.id) in score_mapping

    @pytest.mark.asyncio
    async def test_dlq_score_is_current_timestamp(self, session, redis_client):
        """Req 10.1: DLQ score is failure timestamp in milliseconds."""
        job = FakeJob(retry_count=3, max_retries=3)
        before = int(datetime.now(timezone.utc).replace(tzinfo=None).timestamp() * 1000)

        await handle_job_failure(job, "final failure", session, redis_client)

        after = int(datetime.now(timezone.utc).replace(tzinfo=None).timestamp() * 1000)
        score_mapping = redis_client.zadd.call_args[0][1]
        score = score_mapping[str(job.id)]
        assert before <= score <= after

    @pytest.mark.asyncio
    async def test_clears_worker_id_on_dlq(self, session, redis_client):
        """DLQ jobs clear worker assignment."""
        job = FakeJob(retry_count=3, max_retries=3)

        await handle_job_failure(job, "error", session, redis_client)

        assert job.worker_id is None

    @pytest.mark.asyncio
    async def test_stores_error_on_dlq(self, session, redis_client):
        """DLQ jobs preserve the error message."""
        job = FakeJob(retry_count=3, max_retries=3)
        error_msg = "Permanent failure: invalid data format"

        await handle_job_failure(job, error_msg, session, redis_client)

        assert job.error == error_msg

    @pytest.mark.asyncio
    async def test_dlq_when_max_retries_zero(self, session, redis_client):
        """Req 9.5: Job with max_retries=0 goes directly to DLQ."""
        job = FakeJob(retry_count=0, max_retries=0)

        await handle_job_failure(job, "no retries allowed", session, redis_client)

        assert job.status == JobStatus.DEAD_LETTER
        assert redis_client.zadd.call_args[0][0] == DLQ_KEY

    @pytest.mark.asyncio
    async def test_records_execution_history_on_dlq(self, session, redis_client):
        """DLQ routing still records execution history."""
        job = FakeJob(retry_count=3, max_retries=3)

        await handle_job_failure(job, "error", session, redis_client)

        # At least 2 session.add calls: job + execution record
        assert session.add.call_count >= 2


class TestHandleJobFailureBackoff:
    """Tests for backoff calculation integration."""

    @pytest.fixture
    def session(self):
        mock = AsyncMock()
        mock.add = MagicMock()
        mock.flush = AsyncMock()
        return mock

    @pytest.fixture
    def redis_client(self):
        mock = AsyncMock()
        mock.zadd = AsyncMock()
        return mock

    @pytest.mark.asyncio
    async def test_backoff_increases_with_retry_count(self, session, redis_client):
        """Higher retry counts produce longer backoff delays."""
        job1 = FakeJob(retry_count=0, max_retries=5)
        job2 = FakeJob(retry_count=2, max_retries=5)

        await handle_job_failure(job1, "error", session, redis_client)
        score1 = redis_client.zadd.call_args[0][1][str(job1.id)]

        redis_client.zadd.reset_mock()

        await handle_job_failure(job2, "error", session, redis_client)
        score2 = redis_client.zadd.call_args[0][1][str(job2.id)]

        # Job2 has more retries so its retry time should be further in the future
        assert score2 > score1

    @pytest.mark.asyncio
    async def test_uses_job_backoff_base(self, session, redis_client):
        """Backoff uses the job's configured retry_backoff_base."""
        job_fast = FakeJob(retry_count=0, max_retries=3, retry_backoff_base=1.5)
        job_slow = FakeJob(retry_count=0, max_retries=3, retry_backoff_base=5.0)

        await handle_job_failure(job_fast, "error", session, redis_client)
        score_fast = redis_client.zadd.call_args[0][1][str(job_fast.id)]

        redis_client.zadd.reset_mock()

        await handle_job_failure(job_slow, "error", session, redis_client)
        score_slow = redis_client.zadd.call_args[0][1][str(job_slow.id)]

        # Higher base = longer delay = higher score
        assert score_slow > score_fast


class TestRecordExecution:
    """Tests for the execution history recording helper."""

    @pytest.fixture
    def session(self):
        mock = AsyncMock()
        mock.add = MagicMock()
        return mock

    @pytest.mark.asyncio
    async def test_creates_execution_record(self, session):
        """Records an execution with all required fields."""
        job_id = uuid.uuid4()
        worker_id = uuid.uuid4()
        started = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(seconds=5)
        completed = datetime.now(timezone.utc).replace(tzinfo=None)

        await _record_execution(
            session=session,
            job_id=job_id,
            worker_id=worker_id,
            attempt_number=1,
            status="FAILED",
            started_at=started,
            completed_at=completed,
            error="timeout",
        )

        session.add.assert_called_once()
        execution = session.add.call_args[0][0]
        assert execution.job_id == job_id
        assert execution.worker_id == worker_id
        assert execution.attempt_number == 1
        assert execution.status == "FAILED"
        assert execution.started_at == started
        assert execution.completed_at == completed
        assert execution.error == "timeout"
        assert execution.duration_ms == 5000

    @pytest.mark.asyncio
    async def test_calculates_duration_ms(self, session):
        """Duration is calculated from started_at to completed_at."""
        started = datetime(2024, 1, 1, 12, 0, 0)
        completed = datetime(2024, 1, 1, 12, 0, 2, 500000)  # 2.5 seconds later

        await _record_execution(
            session=session,
            job_id=uuid.uuid4(),
            worker_id=uuid.uuid4(),
            attempt_number=2,
            status="FAILED",
            started_at=started,
            completed_at=completed,
        )

        execution = session.add.call_args[0][0]
        assert execution.duration_ms == 2500

    @pytest.mark.asyncio
    async def test_error_is_optional(self, session):
        """Execution records can be created without an error message."""
        await _record_execution(
            session=session,
            job_id=uuid.uuid4(),
            worker_id=uuid.uuid4(),
            attempt_number=1,
            status="COMPLETED",
            started_at=datetime.now(timezone.utc).replace(tzinfo=None),
            completed_at=datetime.now(timezone.utc).replace(tzinfo=None),
        )

        execution = session.add.call_args[0][0]
        assert execution.error is None
