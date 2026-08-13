"""Tests for PostgreSQL failure handling (Req 15.3, 15.6).

Tests cover:
- API returns 503 for all endpoints when PG is down
- Workers stop dequeuing new jobs when PG is down
- Currently executing jobs complete and hold results in memory
- Pending results are persisted when PG connectivity restores
"""

import asyncio
from datetime import datetime, timezone
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
from starlette.testclient import TestClient

from src.api.health import check_pg_health
from src.api.middleware import (
    PostgresHealthMiddleware,
    _pg_health_cache,
    reset_pg_health_cache,
)
from src.models.enums import JobStatus
from src.workers.worker import (
    PendingJobResult,
    PG_FLUSH_INTERVAL,
    PG_RECONNECT_BACKOFF_START,
    Worker,
)


# --- API Layer Tests ---


class TestPgHealthCheck:
    """Tests for the PostgreSQL health check utility."""

    @pytest.mark.asyncio
    async def test_pg_health_returns_true_when_healthy(self):
        """check_pg_health returns True when SELECT 1 succeeds."""
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()

        mock_engine_connect = AsyncMock()
        mock_engine_connect.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_engine_connect.__aexit__ = AsyncMock(return_value=None)

        with patch("src.api.health.engine") as mock_engine:
            mock_engine.connect.return_value = mock_engine_connect
            result = await check_pg_health()
            assert result is True

    @pytest.mark.asyncio
    async def test_pg_health_returns_false_when_unhealthy(self):
        """check_pg_health returns False when connection fails."""
        with patch("src.api.health.engine") as mock_engine:
            mock_engine.connect.side_effect = Exception("Connection refused")
            result = await check_pg_health()
            assert result is False


class TestPostgresHealthMiddleware:
    """Tests for the PostgreSQL health middleware."""

    def setup_method(self):
        """Reset PG health cache before each test."""
        reset_pg_health_cache()

    @pytest.mark.asyncio
    async def test_middleware_returns_503_when_pg_down(self):
        """Middleware returns 503 for all endpoints when PG is unreachable."""
        from fastapi import FastAPI
        from httpx import ASGITransport, AsyncClient

        app = FastAPI()
        app.add_middleware(PostgresHealthMiddleware)

        @app.get("/test")
        async def test_endpoint():
            return {"status": "ok"}

        @app.post("/jobs")
        async def submit_job():
            return {"id": "123"}

        # Force cache to show PG as down (override the autouse fixture)
        _pg_health_cache["healthy"] = False
        _pg_health_cache["last_check"] = 9999999999.0  # far future to prevent re-check

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/test")
            assert resp.status_code == 503
            assert "PostgreSQL" in resp.json()["detail"]

            resp2 = await client.post("/jobs")
            assert resp2.status_code == 503

    @pytest.mark.asyncio
    async def test_middleware_passes_through_when_pg_healthy(self):
        """Middleware allows requests through when PG is healthy."""
        from fastapi import FastAPI
        from httpx import ASGITransport, AsyncClient

        app = FastAPI()
        app.add_middleware(PostgresHealthMiddleware)

        @app.get("/test")
        async def test_endpoint():
            return {"status": "ok"}

        # Force cache to show PG as healthy (already set by autouse fixture but be explicit)
        _pg_health_cache["healthy"] = True
        _pg_health_cache["last_check"] = 9999999999.0  # far future to prevent re-check

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/test")
            assert resp.status_code == 200
            assert resp.json() == {"status": "ok"}

    @pytest.mark.asyncio
    async def test_middleware_503_for_all_http_methods(self):
        """All HTTP methods return 503 when PG is down."""
        from fastapi import FastAPI
        from httpx import ASGITransport, AsyncClient

        app = FastAPI()
        app.add_middleware(PostgresHealthMiddleware)

        @app.get("/resource")
        async def get_resource():
            return {}

        @app.post("/resource")
        async def create_resource():
            return {}

        @app.delete("/resource/{id}")
        async def delete_resource(id: str):
            return {}

        # Override autouse fixture cache
        _pg_health_cache["healthy"] = False
        _pg_health_cache["last_check"] = 9999999999.0

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            assert (await client.get("/resource")).status_code == 503
            assert (await client.post("/resource")).status_code == 503
            assert (await client.delete("/resource/123")).status_code == 503


# --- Worker Layer Tests ---


class TestWorkerPgFailureHandling:
    """Tests for worker PostgreSQL failure handling behavior."""

    def _create_worker(self) -> Worker:
        """Create a Worker instance with mocked dependencies."""
        worker_id = uuid4()
        session_factory = AsyncMock()
        redis_client = AsyncMock()
        priority_queue = AsyncMock()
        distributed_lock = AsyncMock()
        handler_registry = MagicMock()

        worker = Worker(
            worker_id=worker_id,
            session_factory=session_factory,
            redis_client=redis_client,
            priority_queue=priority_queue,
            distributed_lock=distributed_lock,
            handler_registry=handler_registry,
        )
        return worker

    def test_worker_has_pending_results_list(self):
        """Worker initializes with empty pending_results list."""
        worker = self._create_worker()
        assert worker._pending_results == []
        assert worker._pg_available is True

    def test_pending_job_result_dataclass(self):
        """PendingJobResult holds all required fields."""
        job_id = uuid4()
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        pending = PendingJobResult(
            job_id=job_id,
            status=JobStatus.COMPLETED,
            result={"output": "success"},
            started_at=now,
            completed_at=now,
            worker_id=uuid4(),
            attempt_number=1,
        )
        assert pending.job_id == job_id
        assert pending.status == JobStatus.COMPLETED
        assert pending.result == {"output": "success"}

    @pytest.mark.asyncio
    async def test_worker_stops_dequeuing_when_pg_down(self):
        """Worker pauses polling when _pg_available is False (Req 15.3)."""
        worker = self._create_worker()
        worker._pg_available = False

        # Mock _check_pg_health to always return False
        check_count = 0

        async def mock_check():
            nonlocal check_count
            check_count += 1
            if check_count >= 3:
                worker._shutdown_requested = True
            return False

        worker._check_pg_health = mock_check

        # Run the poll loop briefly — it should not call dequeue
        await worker._poll_loop()

        # The priority queue's dequeue should NOT have been called
        worker._priority_queue.dequeue.assert_not_called()
        # _check_pg_health was called multiple times
        assert check_count >= 2

    @pytest.mark.asyncio
    async def test_worker_resumes_dequeuing_when_pg_recovers(self):
        """Worker resumes dequeuing after PG connectivity restores (Req 15.6)."""
        worker = self._create_worker()
        worker._pg_available = False

        call_count = 0

        async def mock_check():
            nonlocal call_count
            call_count += 1
            # PG recovers on second check
            if call_count >= 2:
                return True
            return False

        worker._check_pg_health = mock_check

        # Make dequeue return None then trigger shutdown
        dequeue_called = False

        async def mock_dequeue(timeout=5.0):
            nonlocal dequeue_called
            dequeue_called = True
            worker._shutdown_requested = True
            return None

        worker._priority_queue.dequeue = mock_dequeue

        await worker._poll_loop()

        assert worker._pg_available is True
        assert dequeue_called is True

    @pytest.mark.asyncio
    async def test_worker_holds_result_in_memory_on_pg_failure(self):
        """Worker stores result in _pending_results when PG fails during persistence (Req 15.3)."""
        worker = self._create_worker()
        job_id = uuid4()

        # Mock job
        mock_job = MagicMock()
        mock_job.id = job_id
        mock_job.timeout_seconds = 300
        mock_job.job_type = "test_job"
        mock_job.payload = {"data": "test"}
        mock_job.retry_count = 0
        mock_job.status = JobStatus.QUEUED

        # Track session factory calls
        call_count = 0

        def make_session_ctx():
            """Create a session context manager for each call."""
            nonlocal call_count
            call_count += 1
            current_call = call_count

            session = AsyncMock()
            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = mock_job
            session.execute = AsyncMock(return_value=mock_result)
            session.add = MagicMock()

            if current_call <= 2:
                # Calls 1-2 succeed (fetch job for lock TTL, update to RUNNING)
                session.begin = MagicMock(return_value=AsyncMock(
                    __aenter__=AsyncMock(return_value=None),
                    __aexit__=AsyncMock(return_value=None),
                ))
            else:
                # Call 3: result persistence fails
                async def raise_on_enter():
                    raise Exception("PG connection lost")

                session.begin = MagicMock(return_value=AsyncMock(
                    __aenter__=AsyncMock(side_effect=Exception("PG connection lost")),
                    __aexit__=AsyncMock(return_value=None),
                ))

            ctx = AsyncMock()
            ctx.__aenter__ = AsyncMock(return_value=session)
            ctx.__aexit__ = AsyncMock(return_value=None)
            return ctx

        worker._session_factory = make_session_ctx

        # Mock lock acquisition success
        worker._distributed_lock.acquire_lock = AsyncMock(return_value=True)
        worker._distributed_lock.release_lock = AsyncMock()

        # Mock handler
        async def mock_handler(payload):
            return {"output": "done"}

        worker._handler_registry.get = MagicMock(return_value=mock_handler)

        await worker.execute_job(job_id)

        # Result should be held in memory
        assert len(worker._pending_results) == 1
        assert worker._pending_results[0].job_id == job_id
        assert worker._pending_results[0].status == JobStatus.COMPLETED
        assert worker._pending_results[0].result == {"output": "done"}
        assert worker._pg_available is False

    @pytest.mark.asyncio
    async def test_flush_pending_results_on_pg_recovery(self):
        """Worker flushes held results when PG connectivity restores (Req 15.6)."""
        worker = self._create_worker()
        job_id = uuid4()

        # Add a pending result
        pending = PendingJobResult(
            job_id=job_id,
            status=JobStatus.COMPLETED,
            result={"output": "done"},
            started_at=datetime.now(timezone.utc).replace(tzinfo=None),
            completed_at=datetime.now(timezone.utc).replace(tzinfo=None),
            worker_id=worker.worker_id,
            attempt_number=1,
        )
        worker._pending_results.append(pending)
        worker._pg_available = False

        # Mock session factory for flush
        mock_job = MagicMock()
        mock_job.id = job_id
        mock_job.status = JobStatus.RUNNING
        mock_job.retry_count = 0

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_job
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.add = MagicMock()

        mock_begin = AsyncMock()
        mock_begin.__aenter__ = AsyncMock(return_value=None)
        mock_begin.__aexit__ = AsyncMock(return_value=None)
        mock_session.begin = MagicMock(return_value=mock_begin)

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_ctx.__aexit__ = AsyncMock(return_value=None)
        worker._session_factory = MagicMock(return_value=mock_ctx)

        await worker._flush_pending_results()

        # Pending results should be cleared
        assert len(worker._pending_results) == 0
        assert worker._pg_available is True

    @pytest.mark.asyncio
    async def test_check_pg_health_returns_true_on_success(self):
        """_check_pg_health returns True when query succeeds."""
        worker = self._create_worker()

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock()

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_ctx.__aexit__ = AsyncMock(return_value=None)
        worker._session_factory = MagicMock(return_value=mock_ctx)

        result = await worker._check_pg_health()
        assert result is True

    @pytest.mark.asyncio
    async def test_check_pg_health_returns_false_on_failure(self):
        """_check_pg_health returns False when query fails."""
        worker = self._create_worker()

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(side_effect=Exception("connection refused"))
        mock_ctx.__aexit__ = AsyncMock(return_value=None)
        worker._session_factory = MagicMock(return_value=mock_ctx)

        result = await worker._check_pg_health()
        assert result is False

    @pytest.mark.asyncio
    async def test_flush_loop_skips_when_no_pending(self):
        """_pg_flush_loop does nothing when _pending_results is empty."""
        worker = self._create_worker()
        worker._pending_results = []

        # Run one iteration of the flush loop then stop
        iteration = 0

        original_sleep = asyncio.sleep

        async def limited_sleep(duration):
            nonlocal iteration
            iteration += 1
            if iteration >= 2:
                worker._shutdown_requested = True
            await original_sleep(0)

        with patch("asyncio.sleep", limited_sleep):
            await worker._pg_flush_loop()

        # No flush should have been attempted
        # _check_pg_health should not have been called
        assert worker._pg_available is True

    @pytest.mark.asyncio
    async def test_flush_stops_on_repeated_pg_failure(self):
        """_flush_pending_results stops flushing when PG fails again."""
        worker = self._create_worker()
        job_id_1 = uuid4()
        job_id_2 = uuid4()

        worker._pending_results = [
            PendingJobResult(
                job_id=job_id_1,
                status=JobStatus.COMPLETED,
                result={"a": 1},
                started_at=datetime.now(timezone.utc).replace(tzinfo=None),
                completed_at=datetime.now(timezone.utc).replace(tzinfo=None),
                worker_id=worker.worker_id,
                attempt_number=1,
            ),
            PendingJobResult(
                job_id=job_id_2,
                status=JobStatus.COMPLETED,
                result={"b": 2},
                started_at=datetime.now(timezone.utc).replace(tzinfo=None),
                completed_at=datetime.now(timezone.utc).replace(tzinfo=None),
                worker_id=worker.worker_id,
                attempt_number=1,
            ),
        ]
        worker._pg_available = False

        # First flush succeeds, second fails
        call_count = 0

        def mock_session_factory():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call succeeds
                mock_job = MagicMock()
                mock_job.id = job_id_1
                mock_job.status = JobStatus.RUNNING
                mock_job.retry_count = 0

                mock_session = AsyncMock()
                mock_result = MagicMock()
                mock_result.scalar_one_or_none.return_value = mock_job
                mock_session.execute = AsyncMock(return_value=mock_result)
                mock_session.add = MagicMock()

                mock_begin = AsyncMock()
                mock_begin.__aenter__ = AsyncMock(return_value=None)
                mock_begin.__aexit__ = AsyncMock(return_value=None)
                mock_session.begin = MagicMock(return_value=mock_begin)

                mock_ctx = AsyncMock()
                mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
                mock_ctx.__aexit__ = AsyncMock(return_value=None)
                return mock_ctx
            else:
                # Second call fails
                mock_ctx = AsyncMock()
                mock_ctx.__aenter__ = AsyncMock(
                    side_effect=Exception("PG down again")
                )
                mock_ctx.__aexit__ = AsyncMock(return_value=None)
                return mock_ctx

        worker._session_factory = mock_session_factory

        await worker._flush_pending_results()

        # First was flushed, second remains
        assert len(worker._pending_results) == 1
        assert worker._pending_results[0].job_id == job_id_2
        # PG is still not available since not all were flushed
        assert worker._pg_available is False
