"""Tests for the GET /metrics endpoint."""

import uuid
from unittest.mock import AsyncMock, MagicMock

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
async def test_get_metrics_default_values(client, mock_coordinator):
    """GET /metrics returns all required fields with default zeros when no activity."""
    mock_coordinator.get_metrics.return_value = {
        "queue_depth": 0,
        "active_workers": 0,
        "jobs_per_second": 0.0,
        "latency_p50_ms": 0.0,
        "latency_p95_ms": 0.0,
        "dlq_size": 0,
    }

    response = await client.get("/metrics")

    assert response.status_code == 200
    data = response.json()
    assert data["queue_depth"] == 0
    assert data["active_workers"] == 0
    assert data["jobs_per_second"] == 0.0
    assert data["latency_p50_ms"] == 0.0
    assert data["latency_p95_ms"] == 0.0
    assert data["dlq_size"] == 0


@pytest.mark.asyncio
async def test_get_metrics_with_activity(client, mock_coordinator):
    """GET /metrics returns non-zero values when system has activity."""
    mock_coordinator.get_metrics.return_value = {
        "queue_depth": 15,
        "active_workers": 3,
        "jobs_per_second": 0.0,
        "latency_p50_ms": 0.0,
        "latency_p95_ms": 0.0,
        "dlq_size": 2,
    }

    response = await client.get("/metrics")

    assert response.status_code == 200
    data = response.json()
    assert data["queue_depth"] == 15
    assert data["active_workers"] == 3
    assert data["dlq_size"] == 2


@pytest.mark.asyncio
async def test_get_metrics_with_collector(app, mock_coordinator):
    """GET /metrics overlays metrics_collector data when available."""
    mock_coordinator.get_metrics.return_value = {
        "queue_depth": 5,
        "active_workers": 2,
        "jobs_per_second": 0.0,
        "latency_p50_ms": 0.0,
        "latency_p95_ms": 0.0,
        "dlq_size": 1,
    }

    # Set up a metrics collector on app state
    metrics_collector = MagicMock()
    metrics_collector.get_current_metrics.return_value = {
        "jobs_per_second": 12.5,
        "latency_p50_ms": 45.0,
        "latency_p95_ms": 150.0,
    }

    app.state.coordinator = mock_coordinator
    app.state.metrics_collector = metrics_collector

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/metrics")

    assert response.status_code == 200
    data = response.json()
    assert data["queue_depth"] == 5
    assert data["active_workers"] == 2
    assert data["jobs_per_second"] == 12.5
    assert data["latency_p50_ms"] == 45.0
    assert data["latency_p95_ms"] == 150.0
    assert data["dlq_size"] == 1


@pytest.mark.asyncio
async def test_get_metrics_all_fields_present(client, mock_coordinator):
    """GET /metrics response always contains all required fields."""
    mock_coordinator.get_metrics.return_value = {
        "queue_depth": 0,
        "active_workers": 0,
        "jobs_per_second": 0.0,
        "latency_p50_ms": 0.0,
        "latency_p95_ms": 0.0,
        "dlq_size": 0,
    }

    response = await client.get("/metrics")

    assert response.status_code == 200
    data = response.json()
    required_fields = [
        "queue_depth",
        "active_workers",
        "jobs_per_second",
        "latency_p50_ms",
        "latency_p95_ms",
        "dlq_size",
    ]
    for field in required_fields:
        assert field in data, f"Missing required field: {field}"
