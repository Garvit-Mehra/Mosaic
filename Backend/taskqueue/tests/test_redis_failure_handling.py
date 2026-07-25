"""Tests for Redis failure handling and reconnection.

Validates:
- Req 15.1: Return 503 for queue-dependent operations when Redis is down,
  continue serving PG-backed reads.
- Req 15.2: Worker pauses polling with exponential backoff (1s start, 30s cap).
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
import redis.asyncio as aioredis
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.dependencies import get_coordinator, require_redis
from src.api.routes.jobs import router as jobs_router
from src.workers.worker import (
    REDIS_RECONNECT_BACKOFF_CAP,
    REDIS_RECONNECT_BACKOFF_START,
    Worker,
)


# ─────────────────────────────────────────────────────────────────────────────
# API Tests: Req 15.1
# ─────────────────────────────────────────────────────────────────────────────


def _create_test_app(redis_healthy: bool = True) -> FastAPI:
    """Create a test FastAPI app with mocked dependencies."""
    app = FastAPI()
    app.include_router(jobs_router)

    mock_coordinator = AsyncMock()
    mock_coordinator.submit_job = AsyncMock(
        return_value={
            "id": str(uuid4()),
            "status": "QUEUED",
            "job_type": "test",
            "priority": 0,
            "created_at": "2024-01-01T00:00:00",
        }
    )
    mock_coordinator.cancel_job = AsyncMock(
        return_value={"message": "cancelled", "job_id": str(uuid4())}
    )
    mock_coordinator.get_job = AsyncMock(
        return_value={
            "id": str(uuid4()),
            "status": "QUEUED",
            "job_type": "test",
            "priority": 0,
        }
    )
    mock_coordinator.list_jobs = AsyncMock(return_value=[])
    mock_coordinator.get_job_history = AsyncMock(return_value=[])

    app.dependency_overrides[get_coordinator] = lambda: mock_coordinator

    if not redis_healthy:
        async def _redis_down():
            from fastapi import HTTPException
            raise HTTPException(
                status_code=503,
                detail="Service temporarily unavailable: queue backend is unreachable",
            )
        app.dependency_overrides[require_redis] = _redis_down
    else:
        # Override with a no-op so tests don't require a live Redis
        async def _redis_ok():
            return None
        app.dependency_overrides[require_redis] = _redis_ok

    return app


class TestApiRedisDown:
    """Test that queue-dependent endpoints return 503 when Redis is down."""

    def test_post_jobs_returns_503_when_redis_down(self):
        """POST /jobs returns 503 when Redis is unreachable (Req 15.1)."""
        app = _create_test_app(redis_healthy=False)
        client = TestClient(app)

        response = client.post(
            "/jobs",
            json={
                "job_type": "test",
                "payload": {"key": "value"},
                "priority": 1,
            },
        )
        assert response.status_code == 503

    def test_delete_jobs_returns_503_when_redis_down(self):
        """DELETE /jobs/:id returns 503 when Redis is unreachable (Req 15.1)."""
        app = _create_test_app(redis_healthy=False)
        client = TestClient(app)

        job_id = str(uuid4())
        response = client.delete(f"/jobs/{job_id}")
        assert response.status_code == 503

    def test_get_jobs_still_works_when_redis_down(self):
        """GET /jobs continues serving when Redis is down (Req 15.1)."""
        app = _create_test_app(redis_healthy=False)
        client = TestClient(app)

        response = client.get("/jobs")
        assert response.status_code == 200

    def test_get_job_by_id_still_works_when_redis_down(self):
        """GET /jobs/:id continues serving when Redis is down (Req 15.1)."""
        app = _create_test_app(redis_healthy=False)
        client = TestClient(app)

        job_id = str(uuid4())
        response = client.get(f"/jobs/{job_id}")
        assert response.status_code == 200

    def test_get_job_history_still_works_when_redis_down(self):
        """GET /jobs/:id/history continues serving when Redis is down (Req 15.1)."""
        app = _create_test_app(redis_healthy=False)
        client = TestClient(app)

        job_id = str(uuid4())
        response = client.get(f"/jobs/{job_id}/history")
        assert response.status_code == 200


class TestApiRedisHealthy:
    """Test that queue-dependent endpoints work normally when Redis is up."""

    def test_post_jobs_works_when_redis_healthy(self):
        """POST /jobs succeeds when Redis is reachable."""
        app = _create_test_app(redis_healthy=True)
        client = TestClient(app)

        response = client.post(
            "/jobs",
            json={
                "job_type": "test",
                "payload": {"key": "value"},
                "priority": 1,
            },
        )
        assert response.status_code == 201

    def test_delete_jobs_works_when_redis_healthy(self):
        """DELETE /jobs/:id works when Redis is reachable."""
        app = _create_test_app(redis_healthy=True)
        client = TestClient(app)

        job_id = str(uuid4())
        response = client.delete(f"/jobs/{job_id}")
        assert response.status_code == 200


# ─────────────────────────────────────────────────────────────────────────────
# Dependency Unit Tests: require_redis
# ─────────────────────────────────────────────────────────────────────────────


class TestRequireRedisDependency:
    """Unit tests for the require_redis dependency."""

    @pytest.mark.asyncio
    async def test_require_redis_passes_when_healthy(self):
        """require_redis does nothing when Redis is reachable."""
        with patch("src.api.dependencies.check_redis_health", return_value=True):
            # Should not raise
            await require_redis()

    @pytest.mark.asyncio
    async def test_require_redis_raises_503_when_unhealthy(self):
        """require_redis raises HTTPException 503 when Redis is down."""
        from fastapi import HTTPException

        with patch("src.api.dependencies.check_redis_health", return_value=False):
            with pytest.raises(HTTPException) as exc_info:
                await require_redis()
            assert exc_info.value.status_code == 503


# ─────────────────────────────────────────────────────────────────────────────
# Worker Tests: Req 15.2
# ─────────────────────────────────────────────────────────────────────────────


class TestWorkerRedisReconnection:
    """Test worker exponential backoff on Redis failures."""

    def _make_worker(self, priority_queue_mock) -> Worker:
        """Create a Worker with mocked dependencies."""
        return Worker(
            worker_id=uuid4(),
            session_factory=MagicMock(),
            redis_client=MagicMock(),
            priority_queue=priority_queue_mock,
            distributed_lock=MagicMock(),
            handler_registry=MagicMock(),
        )

    @pytest.mark.asyncio
    async def test_worker_pauses_on_redis_connection_error(self):
        """Worker sleeps with backoff when dequeue raises ConnectionError (Req 15.2)."""
        mock_queue = AsyncMock()
        # First call raises ConnectionError, second call we shut down
        call_count = 0

        async def dequeue_side_effect(timeout=5.0):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise aioredis.ConnectionError("Connection refused")
            # Signal shutdown to stop the loop
            worker._shutdown_requested = True
            return None

        mock_queue.dequeue = AsyncMock(side_effect=dequeue_side_effect)
        worker = self._make_worker(mock_queue)

        with patch("src.workers.worker.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await worker._poll_loop()

        # Should have slept with initial backoff of 1s
        mock_sleep.assert_called_once_with(REDIS_RECONNECT_BACKOFF_START)

    @pytest.mark.asyncio
    async def test_worker_exponential_backoff_doubles(self):
        """Worker doubles backoff on successive Redis failures (Req 15.2)."""
        mock_queue = AsyncMock()
        call_count = 0

        async def dequeue_side_effect(timeout=5.0):
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                raise aioredis.ConnectionError("Connection refused")
            worker._shutdown_requested = True
            return None

        mock_queue.dequeue = AsyncMock(side_effect=dequeue_side_effect)
        worker = self._make_worker(mock_queue)

        sleep_values = []

        async def capture_sleep(duration):
            sleep_values.append(duration)

        with patch("src.workers.worker.asyncio.sleep", side_effect=capture_sleep):
            await worker._poll_loop()

        # Backoff should be: 1s, 2s, 4s
        assert sleep_values[0] == 1.0
        assert sleep_values[1] == 2.0
        assert sleep_values[2] == 4.0

    @pytest.mark.asyncio
    async def test_worker_backoff_caps_at_30_seconds(self):
        """Worker backoff is capped at 30 seconds (Req 15.2)."""
        mock_queue = AsyncMock()
        call_count = 0

        async def dequeue_side_effect(timeout=5.0):
            nonlocal call_count
            call_count += 1
            if call_count <= 7:
                # 1, 2, 4, 8, 16, 32 -> capped at 30, 30
                raise aioredis.ConnectionError("Connection refused")
            worker._shutdown_requested = True
            return None

        mock_queue.dequeue = AsyncMock(side_effect=dequeue_side_effect)
        worker = self._make_worker(mock_queue)

        sleep_values = []

        async def capture_sleep(duration):
            sleep_values.append(duration)

        with patch("src.workers.worker.asyncio.sleep", side_effect=capture_sleep):
            await worker._poll_loop()

        # Verify cap: after 1, 2, 4, 8, 16 the next would be 32 but capped at 30
        assert sleep_values[5] == REDIS_RECONNECT_BACKOFF_CAP  # 6th failure
        assert sleep_values[6] == REDIS_RECONNECT_BACKOFF_CAP  # 7th failure

    @pytest.mark.asyncio
    async def test_worker_resets_backoff_on_success(self):
        """Worker resets backoff to 1s after successful dequeue (Req 15.2)."""
        mock_queue = AsyncMock()
        call_count = 0

        async def dequeue_side_effect(timeout=5.0):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise aioredis.ConnectionError("Connection refused")
            if call_count == 2:
                # Success (no job, but no exception)
                return None
            if call_count == 3:
                # Another failure — backoff should be reset to 1s
                raise aioredis.ConnectionError("Connection refused")
            worker._shutdown_requested = True
            return None

        mock_queue.dequeue = AsyncMock(side_effect=dequeue_side_effect)
        worker = self._make_worker(mock_queue)

        sleep_values = []

        async def capture_sleep(duration):
            sleep_values.append(duration)

        with patch("src.workers.worker.asyncio.sleep", side_effect=capture_sleep):
            await worker._poll_loop()

        # First failure: sleep 1s, then success resets, then failure again: sleep 1s
        assert sleep_values[0] == 1.0
        assert sleep_values[1] == 1.0  # Reset after success

    @pytest.mark.asyncio
    async def test_worker_handles_timeout_error(self):
        """Worker treats Redis TimeoutError same as ConnectionError (Req 15.2)."""
        mock_queue = AsyncMock()
        call_count = 0

        async def dequeue_side_effect(timeout=5.0):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise aioredis.TimeoutError("Timed out")
            worker._shutdown_requested = True
            return None

        mock_queue.dequeue = AsyncMock(side_effect=dequeue_side_effect)
        worker = self._make_worker(mock_queue)

        with patch("src.workers.worker.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await worker._poll_loop()

        mock_sleep.assert_called_once_with(REDIS_RECONNECT_BACKOFF_START)

    @pytest.mark.asyncio
    async def test_worker_handles_os_error(self):
        """Worker treats OSError same as ConnectionError (Req 15.2)."""
        mock_queue = AsyncMock()
        call_count = 0

        async def dequeue_side_effect(timeout=5.0):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise OSError("Network unreachable")
            worker._shutdown_requested = True
            return None

        mock_queue.dequeue = AsyncMock(side_effect=dequeue_side_effect)
        worker = self._make_worker(mock_queue)

        with patch("src.workers.worker.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await worker._poll_loop()

        mock_sleep.assert_called_once_with(REDIS_RECONNECT_BACKOFF_START)
