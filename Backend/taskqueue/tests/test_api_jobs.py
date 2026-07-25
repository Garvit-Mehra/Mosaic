"""Tests for job submission and status API endpoints.

Tests cover:
- POST /jobs: Submit a new job (Req 1.1, 1.8)
- GET /jobs/{job_id}: Get job metadata (Req 2.1, 2.4)
- GET /jobs: List jobs with status filter and pagination (Req 2.3, 2.5)
- GET /jobs/{job_id}/history: Get execution history (Req 2.2, 2.4)
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.api.dependencies import get_coordinator, require_redis
from src.api.routes.jobs import router as jobs_router
from src.core.coordinator import JobCoordinator
from src.core.validators import ValidationError
from src.models.enums import JobStatus


@pytest.fixture
def mock_coordinator():
    """Create a mock JobCoordinator."""
    coordinator = AsyncMock(spec=JobCoordinator)
    return coordinator


@pytest.fixture
def app(mock_coordinator):
    """Create a test FastAPI app with the jobs router and mock coordinator."""
    test_app = FastAPI()
    test_app.include_router(jobs_router)
    # Override the get_coordinator dependency to return our mock
    test_app.dependency_overrides[get_coordinator] = lambda: mock_coordinator
    # Override require_redis so tests don't need a live Redis connection
    test_app.dependency_overrides[require_redis] = lambda: None
    return test_app


@pytest.fixture
async def client(app):
    """Create an async HTTP client for the test app."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


# --- POST /jobs tests ---


class TestSubmitJob:
    """Tests for POST /jobs endpoint."""

    async def test_submit_job_returns_201(self, client, mock_coordinator):
        """Req 1.1: POST /jobs creates job and returns 201 with job_id and status."""
        job_id = uuid.uuid4()
        created_at = datetime.now(timezone.utc)
        mock_coordinator.submit_job.return_value = {
            "id": job_id,
            "status": "QUEUED",
            "job_type": "send_email",
            "priority": 5,
            "created_at": created_at,
        }

        response = await client.post(
            "/jobs",
            json={
                "job_type": "send_email",
                "payload": {"to": "user@example.com"},
                "priority": 5,
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["id"] == str(job_id)
        assert data["status"] == "QUEUED"
        assert data["job_type"] == "send_email"
        assert data["priority"] == 5
        assert "created_at" in data

    async def test_submit_job_response_fields(self, client, mock_coordinator):
        """Req 1.8: Response includes id, status, job_type, priority, created_at."""
        job_id = uuid.uuid4()
        created_at = datetime.now(timezone.utc)
        mock_coordinator.submit_job.return_value = {
            "id": job_id,
            "status": "SCHEDULED",
            "job_type": "generate_report",
            "priority": 0,
            "created_at": created_at,
        }

        response = await client.post(
            "/jobs",
            json={
                "job_type": "generate_report",
                "payload": {"report_id": 42},
                "execute_at": "2099-01-01T00:00:00Z",
            },
        )

        assert response.status_code == 201
        data = response.json()
        # Verify all required fields are present
        assert "id" in data
        assert "status" in data
        assert "job_type" in data
        assert "priority" in data
        assert "created_at" in data

    async def test_submit_job_validation_error_returns_400(self, client, mock_coordinator):
        """Validation errors from coordinator return appropriate HTTP error."""
        mock_coordinator.submit_job.side_effect = ValidationError(
            status_code=400, detail="priority must be between 0 and 10000, got -1"
        )

        response = await client.post(
            "/jobs",
            json={
                "job_type": "send_email",
                "payload": {"to": "user@example.com"},
                "priority": -1,
            },
        )

        assert response.status_code == 400
        assert "priority" in response.json()["detail"]

    async def test_submit_job_payload_too_large_returns_413(self, client, mock_coordinator):
        """Req 1.7: Oversized payload returns 413."""
        mock_coordinator.submit_job.side_effect = ValidationError(
            status_code=413,
            detail="Payload size (2000000 bytes) exceeds the allowed maximum (1048576 bytes)",
        )

        response = await client.post(
            "/jobs",
            json={
                "job_type": "send_email",
                "payload": {"data": "x" * 1000},
            },
        )

        assert response.status_code == 413

    async def test_submit_job_calls_coordinator_with_correct_params(
        self, client, mock_coordinator
    ):
        """Verify coordinator is called with the correct parameters."""
        job_id = uuid.uuid4()
        mock_coordinator.submit_job.return_value = {
            "id": job_id,
            "status": "QUEUED",
            "job_type": "send_email",
            "priority": 3,
            "created_at": datetime.now(timezone.utc),
        }

        await client.post(
            "/jobs",
            json={
                "job_type": "send_email",
                "payload": {"to": "user@example.com"},
                "priority": 3,
                "timeout_seconds": 600,
                "max_retries": 5,
                "retry_backoff_base": 3.0,
            },
        )

        mock_coordinator.submit_job.assert_called_once_with(
            job_type="send_email",
            payload={"to": "user@example.com"},
            priority=3,
            execute_at=None,
            timeout_seconds=600,
            max_retries=5,
            retry_backoff_base=3.0,
        )


# --- GET /jobs/{job_id} tests ---


class TestGetJob:
    """Tests for GET /jobs/{job_id} endpoint."""

    async def test_get_job_returns_metadata(self, client, mock_coordinator):
        """Req 2.1: GET /jobs/:id returns complete metadata."""
        job_id = uuid.uuid4()
        mock_coordinator.get_job.return_value = {
            "id": job_id,
            "job_type": "send_email",
            "status": "COMPLETED",
            "priority": 5,
            "payload": {"to": "user@example.com"},
            "execute_at": None,
            "timeout_seconds": 300,
            "max_retries": 3,
            "retry_count": 0,
            "retry_backoff_base": 2.0,
            "worker_id": None,
            "started_at": None,
            "completed_at": datetime.now(timezone.utc),
            "result": {"sent": True},
            "error": None,
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        }

        response = await client.get(f"/jobs/{job_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(job_id)
        assert data["status"] == "COMPLETED"
        assert data["job_type"] == "send_email"
        assert data["priority"] == 5

    async def test_get_job_not_found_returns_404(self, client, mock_coordinator):
        """Req 2.4: Non-existent job returns 404."""
        job_id = uuid.uuid4()
        mock_coordinator.get_job.return_value = None

        response = await client.get(f"/jobs/{job_id}")

        assert response.status_code == 404
        assert response.json()["detail"] == "Job not found"


# --- GET /jobs tests ---


class TestListJobs:
    """Tests for GET /jobs endpoint."""

    async def test_list_jobs_returns_list(self, client, mock_coordinator):
        """Req 2.3: GET /jobs returns jobs with pagination."""
        job_id = uuid.uuid4()
        mock_coordinator.list_jobs.return_value = [
            {
                "id": job_id,
                "job_type": "send_email",
                "status": "QUEUED",
                "priority": 5,
                "payload": {},
                "execute_at": None,
                "timeout_seconds": 300,
                "max_retries": 3,
                "retry_count": 0,
                "retry_backoff_base": 2.0,
                "worker_id": None,
                "started_at": None,
                "completed_at": None,
                "result": None,
                "error": None,
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
            }
        ]

        response = await client.get("/jobs")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 1

    async def test_list_jobs_with_status_filter(self, client, mock_coordinator):
        """Req 2.3: Status filter is passed to coordinator."""
        mock_coordinator.list_jobs.return_value = []

        response = await client.get("/jobs?status=RUNNING")

        assert response.status_code == 200
        mock_coordinator.list_jobs.assert_called_once_with(
            status="RUNNING", limit=50, offset=0
        )

    async def test_list_jobs_with_pagination(self, client, mock_coordinator):
        """Req 2.3: Pagination parameters are passed correctly."""
        mock_coordinator.list_jobs.return_value = []

        response = await client.get("/jobs?limit=10&offset=20")

        assert response.status_code == 200
        mock_coordinator.list_jobs.assert_called_once_with(
            status=None, limit=10, offset=20
        )

    async def test_list_jobs_invalid_status_returns_400(self, client, mock_coordinator):
        """Req 2.5: Invalid status filter returns 400 with allowed values."""
        response = await client.get("/jobs?status=INVALID_STATUS")

        assert response.status_code == 400
        detail = response.json()["detail"]
        assert "INVALID_STATUS" in detail
        # Should list allowed values
        assert "PENDING" in detail
        assert "QUEUED" in detail

    async def test_list_jobs_limit_max_100(self, client, mock_coordinator):
        """Req 2.3: Limit max is 100 (enforced by Query validation)."""
        response = await client.get("/jobs?limit=200")

        # FastAPI Query validation with le=100 returns 422 for values > 100
        assert response.status_code == 422

    async def test_list_jobs_limit_min_1(self, client, mock_coordinator):
        """Limit minimum is 1 (enforced by Query validation)."""
        response = await client.get("/jobs?limit=0")

        assert response.status_code == 422

    async def test_list_jobs_offset_min_0(self, client, mock_coordinator):
        """Offset must be non-negative."""
        response = await client.get("/jobs?offset=-1")

        assert response.status_code == 422


# --- GET /jobs/{job_id}/history tests ---


class TestGetJobHistory:
    """Tests for GET /jobs/{job_id}/history endpoint."""

    async def test_get_job_history_returns_executions(self, client, mock_coordinator):
        """Req 2.2: Returns execution attempts ordered by attempt_number."""
        job_id = uuid.uuid4()
        worker_id = uuid.uuid4()
        mock_coordinator.get_job.return_value = {
            "id": job_id,
            "status": "COMPLETED",
            "job_type": "send_email",
            "priority": 0,
        }
        mock_coordinator.get_job_history.return_value = [
            {
                "id": uuid.uuid4(),
                "job_id": job_id,
                "worker_id": worker_id,
                "attempt_number": 1,
                "status": "FAILED",
                "started_at": datetime.now(timezone.utc),
                "completed_at": datetime.now(timezone.utc),
                "duration_ms": 150,
                "error": "Connection timeout",
            },
            {
                "id": uuid.uuid4(),
                "job_id": job_id,
                "worker_id": worker_id,
                "attempt_number": 2,
                "status": "COMPLETED",
                "started_at": datetime.now(timezone.utc),
                "completed_at": datetime.now(timezone.utc),
                "duration_ms": 200,
                "error": None,
            },
        ]

        response = await client.get(f"/jobs/{job_id}/history")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]["attempt_number"] == 1
        assert data[1]["attempt_number"] == 2

    async def test_get_job_history_not_found_returns_404(self, client, mock_coordinator):
        """Req 2.4: Non-existent job on history endpoint returns 404."""
        job_id = uuid.uuid4()
        mock_coordinator.get_job.return_value = None

        response = await client.get(f"/jobs/{job_id}/history")

        assert response.status_code == 404
        assert response.json()["detail"] == "Job not found"

    async def test_get_job_history_empty_list(self, client, mock_coordinator):
        """Job with no execution history returns empty list."""
        job_id = uuid.uuid4()
        mock_coordinator.get_job.return_value = {
            "id": job_id,
            "status": "PENDING",
            "job_type": "send_email",
            "priority": 0,
        }
        mock_coordinator.get_job_history.return_value = []

        response = await client.get(f"/jobs/{job_id}/history")

        assert response.status_code == 200
        assert response.json() == []
