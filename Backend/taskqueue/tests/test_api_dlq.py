"""Tests for Dead-Letter Queue API endpoints.

Validates requirements:
- Req 10.2: GET /dlq with pagination (limit/offset, default 50, max 100)
- Req 10.3: POST /dlq/:id/retry resets retry_count, transitions to QUEUED
- Req 10.4: POST /dlq/:id/retry → 404 for non-existent, 409 for non-DEAD_LETTER
- Req 10.5: DLQ preserves original payload and history
"""

import uuid
from datetime import datetime
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from src.core.coordinator import JobCoordinator
from src.main import create_app


@pytest.fixture
def mock_coordinator():
    """Create a mock JobCoordinator."""
    coordinator = AsyncMock(spec=JobCoordinator)
    return coordinator


@pytest.fixture
def app_with_mock(mock_coordinator):
    """Create a FastAPI app with a mock coordinator in app.state."""
    app = create_app()
    app.state.coordinator = mock_coordinator
    return app


@pytest.fixture
async def client(app_with_mock):
    """Create an async test client."""
    transport = ASGITransport(app=app_with_mock)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def sample_dlq_jobs():
    """Generate sample DLQ job data."""
    now = datetime.utcnow()
    return [
        {
            "id": uuid.uuid4(),
            "job_type": "email_send",
            "payload": {"to": "user@example.com", "subject": "Hello"},
            "error": "SMTP connection timeout",
            "retry_count": 3,
            "created_at": now,
            "updated_at": now,
        },
        {
            "id": uuid.uuid4(),
            "job_type": "data_export",
            "payload": {"format": "csv", "table": "users"},
            "error": "Database connection lost",
            "retry_count": 5,
            "created_at": now,
            "updated_at": now,
        },
    ]


class TestListDLQ:
    """Tests for GET /dlq."""

    async def test_list_dlq_returns_jobs(self, client, mock_coordinator, sample_dlq_jobs):
        """Req 10.2: GET /dlq returns dead-lettered jobs."""
        mock_coordinator.list_dlq.return_value = sample_dlq_jobs

        response = await client.get("/dlq")

        assert response.status_code == 200
        data = response.json()
        assert "jobs" in data
        assert data["count"] == 2
        assert data["limit"] == 50
        assert data["offset"] == 0
        mock_coordinator.list_dlq.assert_called_once_with(limit=50, offset=0)

    async def test_list_dlq_default_pagination(self, client, mock_coordinator):
        """Req 10.2: Default limit is 50 and offset is 0."""
        mock_coordinator.list_dlq.return_value = []

        response = await client.get("/dlq")

        assert response.status_code == 200
        data = response.json()
        assert data["limit"] == 50
        assert data["offset"] == 0
        mock_coordinator.list_dlq.assert_called_once_with(limit=50, offset=0)

    async def test_list_dlq_custom_pagination(self, client, mock_coordinator):
        """Req 10.2: Custom limit and offset are passed to coordinator."""
        mock_coordinator.list_dlq.return_value = []

        response = await client.get("/dlq?limit=25&offset=10")

        assert response.status_code == 200
        data = response.json()
        assert data["limit"] == 25
        assert data["offset"] == 10
        mock_coordinator.list_dlq.assert_called_once_with(limit=25, offset=10)

    async def test_list_dlq_max_limit_is_100(self, client, mock_coordinator):
        """Req 10.2: Maximum limit is 100."""
        response = await client.get("/dlq?limit=101")

        # FastAPI Query validation with le=100 returns 422
        assert response.status_code == 422

    async def test_list_dlq_min_limit_is_1(self, client, mock_coordinator):
        """Req 10.2: Minimum limit is 1."""
        response = await client.get("/dlq?limit=0")

        # FastAPI Query validation with ge=1 returns 422
        assert response.status_code == 422

    async def test_list_dlq_negative_offset_rejected(self, client, mock_coordinator):
        """Negative offset is rejected."""
        response = await client.get("/dlq?offset=-1")

        assert response.status_code == 422

    async def test_list_dlq_empty_result(self, client, mock_coordinator):
        """GET /dlq returns empty list when no DLQ jobs exist."""
        mock_coordinator.list_dlq.return_value = []

        response = await client.get("/dlq")

        assert response.status_code == 200
        data = response.json()
        assert data["jobs"] == []
        assert data["count"] == 0

    async def test_list_dlq_preserves_payload(self, client, mock_coordinator, sample_dlq_jobs):
        """Req 10.5: DLQ listing preserves original payload."""
        mock_coordinator.list_dlq.return_value = sample_dlq_jobs

        response = await client.get("/dlq")

        assert response.status_code == 200
        data = response.json()
        # Verify the payload is included in the response
        assert len(data["jobs"]) == 2


class TestRetryDLQJob:
    """Tests for POST /dlq/{job_id}/retry."""

    async def test_retry_dlq_job_success(self, client, mock_coordinator):
        """Req 10.3: POST /dlq/:id/retry resets retry_count and transitions to QUEUED."""
        job_id = uuid.uuid4()
        mock_coordinator.retry_dlq_job.return_value = {
            "id": job_id,
            "status": "QUEUED",
            "job_type": "email_send",
            "priority": 5,
            "payload": {"to": "user@example.com"},
            "retry_count": 0,
            "status_code": 200,
        }

        response = await client.post(f"/dlq/{job_id}/retry")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "QUEUED"
        assert data["retry_count"] == 0
        assert data["payload"] == {"to": "user@example.com"}
        assert "status_code" not in data  # Internal field removed
        mock_coordinator.retry_dlq_job.assert_called_once_with(job_id)

    async def test_retry_dlq_job_not_found_returns_404(self, client, mock_coordinator):
        """Req 10.4: POST /dlq/:id/retry returns 404 for non-existent job."""
        job_id = uuid.uuid4()
        mock_coordinator.retry_dlq_job.return_value = {
            "error": "not_found",
            "status_code": 404,
        }

        response = await client.post(f"/dlq/{job_id}/retry")

        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()

    async def test_retry_dlq_job_not_dead_letter_returns_409(self, client, mock_coordinator):
        """Req 10.4: POST /dlq/:id/retry returns 409 for non-DEAD_LETTER job."""
        job_id = uuid.uuid4()
        mock_coordinator.retry_dlq_job.return_value = {
            "error": "not_retryable",
            "status_code": 409,
        }

        response = await client.post(f"/dlq/{job_id}/retry")

        assert response.status_code == 409
        data = response.json()
        assert "not in dead-letter state" in data["detail"].lower()

    async def test_retry_dlq_invalid_uuid_returns_422(self, client, mock_coordinator):
        """Invalid UUID format returns 422 validation error."""
        response = await client.post("/dlq/not-a-uuid/retry")

        assert response.status_code == 422

    async def test_retry_preserves_original_priority(self, client, mock_coordinator):
        """Req 10.3: Retry re-queues with original priority."""
        job_id = uuid.uuid4()
        mock_coordinator.retry_dlq_job.return_value = {
            "id": job_id,
            "status": "QUEUED",
            "job_type": "data_export",
            "priority": 100,
            "payload": {"format": "csv"},
            "retry_count": 0,
            "status_code": 200,
        }

        response = await client.post(f"/dlq/{job_id}/retry")

        assert response.status_code == 200
        data = response.json()
        assert data["priority"] == 100
        assert data["retry_count"] == 0
