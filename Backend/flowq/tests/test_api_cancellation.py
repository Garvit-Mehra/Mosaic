"""Tests for job cancellation endpoint (DELETE /jobs/:id).

Validates requirements:
- Req 3.1: DELETE /jobs/:id cancels PENDING/SCHEDULED/QUEUED → 200
- Req 3.2: DELETE /jobs/:id for RUNNING/COMPLETED/FAILED/DEAD_LETTER → 409
- Req 3.3: DELETE already-CANCELLED → 200 (idempotent)
- Req 3.4: DELETE non-existent → 404
"""

import uuid
from unittest.mock import AsyncMock, patch

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


class TestCancelJob:
    """Tests for DELETE /jobs/{job_id}."""

    async def test_cancel_pending_job_returns_200(self, client, mock_coordinator):
        """Req 3.1: Cancelling a PENDING job returns 200."""
        job_id = uuid.uuid4()
        mock_coordinator.cancel_job.return_value = {"success": True, "status_code": 200}

        response = await client.delete(f"/jobs/{job_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Job cancelled successfully"
        assert data["job_id"] == str(job_id)
        mock_coordinator.cancel_job.assert_called_once_with(job_id)

    async def test_cancel_scheduled_job_returns_200(self, client, mock_coordinator):
        """Req 3.1: Cancelling a SCHEDULED job returns 200."""
        job_id = uuid.uuid4()
        mock_coordinator.cancel_job.return_value = {"success": True, "status_code": 200}

        response = await client.delete(f"/jobs/{job_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Job cancelled successfully"

    async def test_cancel_queued_job_returns_200(self, client, mock_coordinator):
        """Req 3.1: Cancelling a QUEUED job returns 200."""
        job_id = uuid.uuid4()
        mock_coordinator.cancel_job.return_value = {"success": True, "status_code": 200}

        response = await client.delete(f"/jobs/{job_id}")

        assert response.status_code == 200

    async def test_cancel_running_job_returns_409(self, client, mock_coordinator):
        """Req 3.2: Cancelling a RUNNING job returns 409."""
        job_id = uuid.uuid4()
        mock_coordinator.cancel_job.return_value = {
            "error": "not_cancellable",
            "status_code": 409,
        }

        response = await client.delete(f"/jobs/{job_id}")

        assert response.status_code == 409
        data = response.json()
        assert "not cancellable" in data["detail"]

    async def test_cancel_completed_job_returns_409(self, client, mock_coordinator):
        """Req 3.2: Cancelling a COMPLETED job returns 409."""
        job_id = uuid.uuid4()
        mock_coordinator.cancel_job.return_value = {
            "error": "not_cancellable",
            "status_code": 409,
        }

        response = await client.delete(f"/jobs/{job_id}")

        assert response.status_code == 409

    async def test_cancel_failed_job_returns_409(self, client, mock_coordinator):
        """Req 3.2: Cancelling a FAILED job returns 409."""
        job_id = uuid.uuid4()
        mock_coordinator.cancel_job.return_value = {
            "error": "not_cancellable",
            "status_code": 409,
        }

        response = await client.delete(f"/jobs/{job_id}")

        assert response.status_code == 409

    async def test_cancel_dead_letter_job_returns_409(self, client, mock_coordinator):
        """Req 3.2: Cancelling a DEAD_LETTER job returns 409."""
        job_id = uuid.uuid4()
        mock_coordinator.cancel_job.return_value = {
            "error": "not_cancellable",
            "status_code": 409,
        }

        response = await client.delete(f"/jobs/{job_id}")

        assert response.status_code == 409

    async def test_cancel_already_cancelled_returns_200(self, client, mock_coordinator):
        """Req 3.3: Cancelling an already-CANCELLED job returns 200 (idempotent)."""
        job_id = uuid.uuid4()
        mock_coordinator.cancel_job.return_value = {"success": True, "status_code": 200}

        response = await client.delete(f"/jobs/{job_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Job cancelled successfully"

    async def test_cancel_nonexistent_job_returns_404(self, client, mock_coordinator):
        """Req 3.4: Cancelling a non-existent job returns 404."""
        job_id = uuid.uuid4()
        mock_coordinator.cancel_job.return_value = {
            "error": "not_found",
            "status_code": 404,
        }

        response = await client.delete(f"/jobs/{job_id}")

        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()

    async def test_cancel_invalid_uuid_returns_422(self, client, mock_coordinator):
        """Invalid UUID format returns 422 validation error."""
        response = await client.delete("/jobs/not-a-uuid")

        assert response.status_code == 422
