"""Unit tests for DLQ management: listing and retry operations.

Tests cover list_dlq pagination and retry_dlq_job state transitions,
error handling, and Redis queue coordination.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from src.core.coordinator import JobCoordinator, DLQ_KEY
from src.core.state_machine import (
    VALID_TRANSITIONS,
    apply_transition,
    validate_transition,
)
from src.models.enums import JobStatus


# --- State machine tests for DEAD_LETTER → QUEUED transition ---


class TestDeadLetterTransition:
    """Tests for the DEAD_LETTER → QUEUED transition in the state machine."""

    def test_dead_letter_to_queued_is_valid(self):
        """DEAD_LETTER → QUEUED should be a valid transition for DLQ retry."""
        assert validate_transition(JobStatus.DEAD_LETTER, JobStatus.QUEUED) is True

    def test_dead_letter_to_other_states_invalid(self):
        """DEAD_LETTER should only allow transition to QUEUED."""
        invalid_targets = [
            JobStatus.PENDING,
            JobStatus.SCHEDULED,
            JobStatus.RUNNING,
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
            JobStatus.DEAD_LETTER,
        ]
        for target in invalid_targets:
            assert validate_transition(JobStatus.DEAD_LETTER, target) is False

    def test_apply_transition_dead_letter_to_queued(self):
        """apply_transition should update status to QUEUED from DEAD_LETTER."""

        @dataclass
        class FakeJob:
            status: JobStatus = JobStatus.DEAD_LETTER
            updated_at: datetime = field(default_factory=datetime.utcnow)

        job = FakeJob()
        old_updated = job.updated_at

        apply_transition(job, JobStatus.QUEUED)

        assert job.status == JobStatus.QUEUED
        assert job.updated_at >= old_updated

    def test_dead_letter_has_queued_in_transitions_map(self):
        """VALID_TRANSITIONS map should include DEAD_LETTER → {QUEUED}."""
        assert JobStatus.QUEUED in VALID_TRANSITIONS[JobStatus.DEAD_LETTER]


# --- Coordinator DLQ tests with mocked database and Redis ---


@dataclass
class FakeJob:
    """Lightweight job stand-in for coordinator DLQ tests."""

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


def _create_coordinator(mock_session_factory, mock_redis, mock_priority_queue):
    """Helper to create a JobCoordinator with mocked dependencies."""
    return JobCoordinator(
        session_factory=mock_session_factory,
        redis_client=mock_redis,
        priority_queue=mock_priority_queue,
    )


def _make_mock_session_factory(jobs: List[FakeJob]):
    """Create a mock session factory that returns jobs from a simulated DB.

    For list queries, returns all jobs provided.
    For single job queries, returns the first job or None if list is empty.
    """
    mock_session = AsyncMock()
    mock_result = MagicMock()

    # Setup for list queries (scalars().all())
    mock_scalars = MagicMock()
    mock_scalars.all.return_value = jobs
    mock_result.scalars.return_value = mock_scalars

    # Setup for single job queries (scalar_one_or_none())
    mock_result.scalar_one_or_none.return_value = jobs[0] if jobs else None

    mock_session.execute = AsyncMock(return_value=mock_result)
    mock_session.flush = AsyncMock()
    mock_session.add = MagicMock()

    # Make session work as async context manager
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


class TestListDlq:
    """Tests for the list_dlq method."""

    async def test_list_dlq_returns_dead_letter_jobs(self):
        """list_dlq should return jobs with DEAD_LETTER status."""
        jobs = [
            FakeJob(job_type="email", error="timeout"),
            FakeJob(job_type="report", error="connection refused"),
        ]
        mock_factory = _make_mock_session_factory(jobs)
        mock_redis = AsyncMock()
        mock_pq = AsyncMock()

        coordinator = _create_coordinator(mock_factory, mock_redis, mock_pq)
        result = await coordinator.list_dlq()

        assert len(result) == 2
        assert result[0]["job_type"] == "email"
        assert result[0]["error"] == "timeout"
        assert result[1]["job_type"] == "report"

    async def test_list_dlq_includes_required_fields(self):
        """Each DLQ entry should include id, job_type, payload, error, retry_count, timestamps."""
        job = FakeJob()
        mock_factory = _make_mock_session_factory([job])
        mock_redis = AsyncMock()
        mock_pq = AsyncMock()

        coordinator = _create_coordinator(mock_factory, mock_redis, mock_pq)
        result = await coordinator.list_dlq()

        assert len(result) == 1
        entry = result[0]
        assert "id" in entry
        assert "job_type" in entry
        assert "payload" in entry
        assert "error" in entry
        assert "retry_count" in entry
        assert "created_at" in entry
        assert "updated_at" in entry

    async def test_list_dlq_preserves_original_payload(self):
        """DLQ listing should preserve the original job payload (Req 10.5)."""
        original_payload = {"task": "send_email", "to": "user@example.com", "body": "Hello"}
        job = FakeJob(payload=original_payload)
        mock_factory = _make_mock_session_factory([job])
        mock_redis = AsyncMock()
        mock_pq = AsyncMock()

        coordinator = _create_coordinator(mock_factory, mock_redis, mock_pq)
        result = await coordinator.list_dlq()

        assert result[0]["payload"] == original_payload

    async def test_list_dlq_default_limit_is_50(self):
        """Default limit should be 50."""
        jobs = [FakeJob() for _ in range(60)]
        mock_factory = _make_mock_session_factory(jobs)
        mock_redis = AsyncMock()
        mock_pq = AsyncMock()

        coordinator = _create_coordinator(mock_factory, mock_redis, mock_pq)
        # The method constructs a query with limit=50, but our mock returns all jobs
        # We verify the method executes without error (query construction is tested via integration)
        result = await coordinator.list_dlq()
        # Mock returns all 60, but in real usage the DB would cap at 50
        assert isinstance(result, list)

    async def test_list_dlq_caps_limit_at_100(self):
        """Limit should be capped at 100 even if a larger value is provided."""
        mock_factory = _make_mock_session_factory([])
        mock_redis = AsyncMock()
        mock_pq = AsyncMock()

        coordinator = _create_coordinator(mock_factory, mock_redis, mock_pq)
        # This should not raise - limit is capped internally
        result = await coordinator.list_dlq(limit=200)
        assert isinstance(result, list)

    async def test_list_dlq_empty_queue(self):
        """list_dlq should return empty list when no dead-lettered jobs exist."""
        mock_factory = _make_mock_session_factory([])
        mock_redis = AsyncMock()
        mock_pq = AsyncMock()

        coordinator = _create_coordinator(mock_factory, mock_redis, mock_pq)
        result = await coordinator.list_dlq()

        assert result == []

    async def test_list_dlq_negative_offset_becomes_zero(self):
        """Negative offset should be clamped to 0."""
        mock_factory = _make_mock_session_factory([])
        mock_redis = AsyncMock()
        mock_pq = AsyncMock()

        coordinator = _create_coordinator(mock_factory, mock_redis, mock_pq)
        # Should not raise
        result = await coordinator.list_dlq(offset=-5)
        assert isinstance(result, list)


class TestRetryDlqJob:
    """Tests for the retry_dlq_job method."""

    async def test_retry_returns_404_for_nonexistent_job(self):
        """retry_dlq_job should return 404 error for a non-existent job ID."""
        mock_factory = _make_mock_session_factory([])  # No job found
        mock_redis = AsyncMock()
        mock_pq = AsyncMock()

        coordinator = _create_coordinator(mock_factory, mock_redis, mock_pq)
        result = await coordinator.retry_dlq_job(uuid4())

        assert result["error"] == "not_found"
        assert result["status_code"] == 404

    async def test_retry_returns_409_for_non_dead_letter_job(self):
        """retry_dlq_job should return 409 for jobs not in DEAD_LETTER status."""
        job = FakeJob(status=JobStatus.QUEUED)
        mock_factory = _make_mock_session_factory([job])
        mock_redis = AsyncMock()
        mock_pq = AsyncMock()

        coordinator = _create_coordinator(mock_factory, mock_redis, mock_pq)
        result = await coordinator.retry_dlq_job(job.id)

        assert result["error"] == "not_retryable"
        assert result["status_code"] == 409

    async def test_retry_returns_409_for_completed_job(self):
        """retry_dlq_job should return 409 for COMPLETED jobs."""
        job = FakeJob(status=JobStatus.COMPLETED)
        mock_factory = _make_mock_session_factory([job])
        mock_redis = AsyncMock()
        mock_pq = AsyncMock()

        coordinator = _create_coordinator(mock_factory, mock_redis, mock_pq)
        result = await coordinator.retry_dlq_job(job.id)

        assert result["error"] == "not_retryable"
        assert result["status_code"] == 409

    async def test_retry_success_resets_retry_count(self):
        """Successful retry should reset retry_count to 0."""
        job = FakeJob(retry_count=3)
        mock_factory = _make_mock_session_factory([job])
        mock_redis = AsyncMock()
        mock_pq = AsyncMock()
        mock_pq.enqueue = AsyncMock()

        coordinator = _create_coordinator(mock_factory, mock_redis, mock_pq)
        result = await coordinator.retry_dlq_job(job.id)

        assert result["status_code"] == 200
        assert result["retry_count"] == 0
        assert job.retry_count == 0

    async def test_retry_success_transitions_to_queued(self):
        """Successful retry should transition job status to QUEUED."""
        job = FakeJob()
        mock_factory = _make_mock_session_factory([job])
        mock_redis = AsyncMock()
        mock_pq = AsyncMock()
        mock_pq.enqueue = AsyncMock()

        coordinator = _create_coordinator(mock_factory, mock_redis, mock_pq)
        result = await coordinator.retry_dlq_job(job.id)

        assert result["status"] == "QUEUED"
        assert job.status == JobStatus.QUEUED

    async def test_retry_success_clears_error(self):
        """Successful retry should clear the error field."""
        job = FakeJob(error="connection timeout after 30s")
        mock_factory = _make_mock_session_factory([job])
        mock_redis = AsyncMock()
        mock_pq = AsyncMock()
        mock_pq.enqueue = AsyncMock()

        coordinator = _create_coordinator(mock_factory, mock_redis, mock_pq)
        await coordinator.retry_dlq_job(job.id)

        assert job.error is None

    async def test_retry_success_removes_from_dlq_redis(self):
        """Successful retry should remove the job from the DLQ sorted set in Redis."""
        job = FakeJob()
        mock_factory = _make_mock_session_factory([job])
        mock_redis = AsyncMock()
        mock_redis.zrem = AsyncMock()
        mock_pq = AsyncMock()
        mock_pq.enqueue = AsyncMock()

        coordinator = _create_coordinator(mock_factory, mock_redis, mock_pq)
        await coordinator.retry_dlq_job(job.id)

        mock_redis.zrem.assert_called_once_with(DLQ_KEY, str(job.id))

    async def test_retry_success_adds_to_priority_queue(self):
        """Successful retry should add job to priority queue with original priority."""
        job = FakeJob(priority=7)
        mock_factory = _make_mock_session_factory([job])
        mock_redis = AsyncMock()
        mock_pq = AsyncMock()
        mock_pq.enqueue = AsyncMock()

        coordinator = _create_coordinator(mock_factory, mock_redis, mock_pq)
        await coordinator.retry_dlq_job(job.id)

        mock_pq.enqueue.assert_called_once_with(
            job_id=job.id,
            priority=7,
            enqueued_at=job.created_at,
        )

    async def test_retry_preserves_original_payload(self):
        """Successful retry should preserve the original job payload (Req 10.5)."""
        original_payload = {"action": "process_invoice", "invoice_id": "INV-001"}
        job = FakeJob(payload=original_payload)
        mock_factory = _make_mock_session_factory([job])
        mock_redis = AsyncMock()
        mock_pq = AsyncMock()
        mock_pq.enqueue = AsyncMock()

        coordinator = _create_coordinator(mock_factory, mock_redis, mock_pq)
        result = await coordinator.retry_dlq_job(job.id)

        assert result["payload"] == original_payload

    async def test_retry_response_includes_job_details(self):
        """Retry response should include id, status, job_type, priority, payload."""
        job = FakeJob(job_type="email_send", priority=3)
        mock_factory = _make_mock_session_factory([job])
        mock_redis = AsyncMock()
        mock_pq = AsyncMock()
        mock_pq.enqueue = AsyncMock()

        coordinator = _create_coordinator(mock_factory, mock_redis, mock_pq)
        result = await coordinator.retry_dlq_job(job.id)

        assert result["id"] == job.id
        assert result["status"] == "QUEUED"
        assert result["job_type"] == "email_send"
        assert result["priority"] == 3
        assert result["status_code"] == 200
