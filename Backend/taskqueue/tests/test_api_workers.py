"""Tests for the GET /workers endpoint."""

import uuid
from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from src.main import create_app


@pytest.fixture
def app():
    """Create test FastAPI app with mocked coordinator."""
    test_app = create_app()
    return test_app


@pytest.fixture
def mock_coordinator():
    """Create a mock JobCoordinator."""
    coordinator = AsyncMock()
    return coordinator


@pytest.fixture
async def client(app, mock_coordinator):
    """Create an async test client with mocked coordinator."""
    app.state.coordinator = mock_coordinator
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_list_workers_empty(client, mock_coordinator):
    """GET /workers returns empty list when no workers registered."""
    mock_coordinator.list_workers.return_value = []

    response = await client.get("/workers")

    assert response.status_code == 200
    assert response.json() == []
    mock_coordinator.list_workers.assert_called_once()


@pytest.mark.asyncio
async def test_list_workers_returns_worker_info(client, mock_coordinator):
    """GET /workers returns worker details including id, status, current_job_id, last_heartbeat, jobs_completed."""
    worker_id = uuid.uuid4()
    job_id = uuid.uuid4()
    now = datetime(2024, 1, 15, 12, 0, 0)

    mock_coordinator.list_workers.return_value = [
        {
            "id": worker_id,
            "status": "ACTIVE",
            "current_job_id": job_id,
            "last_heartbeat": now.isoformat(),
            "jobs_completed": 42,
        }
    ]

    response = await client.get("/workers")

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    worker = data[0]
    assert worker["id"] == str(worker_id)
    assert worker["status"] == "ACTIVE"
    assert worker["current_job_id"] == str(job_id)
    assert worker["jobs_completed"] == 42


@pytest.mark.asyncio
async def test_list_workers_multiple_workers(client, mock_coordinator):
    """GET /workers returns multiple workers."""
    worker1_id = uuid.uuid4()
    worker2_id = uuid.uuid4()
    now = datetime(2024, 1, 15, 12, 0, 0)

    mock_coordinator.list_workers.return_value = [
        {
            "id": worker1_id,
            "status": "ACTIVE",
            "current_job_id": None,
            "last_heartbeat": now.isoformat(),
            "jobs_completed": 10,
        },
        {
            "id": worker2_id,
            "status": "IDLE",
            "current_job_id": None,
            "last_heartbeat": now.isoformat(),
            "jobs_completed": 5,
        },
    ]

    response = await client.get("/workers")

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["status"] == "ACTIVE"
    assert data[1]["status"] == "IDLE"
