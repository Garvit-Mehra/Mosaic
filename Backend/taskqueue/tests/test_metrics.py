"""Unit and integration tests for the MetricsCollector.

Tests cover:
- Latency sample recording and retrieval
- Percentile calculations (P50, P95)
- Jobs-per-second throughput calculation
- Zero-value returns when no jobs have been processed
- Sliding window expiry of old samples

Requirements: 14.1, 14.3, 14.5
"""

import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.metrics.collector import (
    DLQ_KEY,
    LATENCY_SAMPLES_KEY,
    WINDOW_MS,
    MetricsCollector,
    _percentile,
)


def _redis_available() -> bool:
    """Check if Redis is reachable on localhost:6379."""
    import socket

    try:
        s = socket.create_connection(("localhost", 6379), timeout=1)
        s.close()
        return True
    except (OSError, socket.timeout):
        return False


requires_redis = pytest.mark.skipif(
    not _redis_available(), reason="Redis not available on localhost:6379"
)


class TestPercentileCalculation:
    """Tests for the pure _percentile function."""

    def test_empty_list_returns_zero(self):
        assert _percentile([], 50) == 0.0
        assert _percentile([], 95) == 0.0

    def test_single_value(self):
        assert _percentile([100.0], 50) == 100.0
        assert _percentile([100.0], 95) == 100.0

    def test_p50_even_count(self):
        """P50 of [10, 20, 30, 40] should interpolate between 20 and 30."""
        values = [10.0, 20.0, 30.0, 40.0]
        p50 = _percentile(values, 50)
        assert p50 == 25.0

    def test_p50_odd_count(self):
        """P50 of [10, 20, 30, 40, 50] should be 30."""
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        p50 = _percentile(values, 50)
        assert p50 == 30.0

    def test_p95_multiple_values(self):
        """P95 should return a value near the high end."""
        values = list(range(1, 101))  # 1 to 100
        p95 = _percentile([float(v) for v in values], 95)
        # With 100 values: rank = 0.95 * 99 = 94.05
        # Interpolate between index 94 (value 95) and index 95 (value 96)
        assert 95.0 <= p95 <= 96.0

    def test_p0_returns_first(self):
        values = [10.0, 20.0, 30.0]
        assert _percentile(values, 0) == 10.0

    def test_p100_returns_last(self):
        values = [10.0, 20.0, 30.0]
        assert _percentile(values, 100) == 30.0

    def test_two_values(self):
        """With two values, P50 should be the midpoint."""
        values = [100.0, 200.0]
        p50 = _percentile(values, 50)
        assert p50 == 150.0


class TestMetricsCollectorUnit:
    """Unit tests for MetricsCollector using mocked dependencies."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client."""
        client = AsyncMock()
        pipe = MagicMock()
        pipe.zadd = MagicMock(return_value=pipe)
        pipe.incr = MagicMock(return_value=pipe)
        pipe.zremrangebyscore = MagicMock(return_value=pipe)
        pipe.execute = AsyncMock(return_value=[1, 1, 0])
        client.pipeline = MagicMock(return_value=pipe)
        return client

    @pytest.fixture
    def mock_session_factory(self):
        """Create a mock session factory that returns zero active workers."""
        session = AsyncMock()
        result = MagicMock()
        result.scalar_one.return_value = 0
        session.execute = AsyncMock(return_value=result)
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=None)

        factory = MagicMock()
        factory.return_value = session
        return factory

    @pytest.fixture
    def mock_priority_queue(self):
        """Create a mock priority queue."""
        queue = AsyncMock()
        queue.depth = AsyncMock(return_value=0)
        return queue

    @pytest.fixture
    def collector(self, mock_redis, mock_session_factory, mock_priority_queue):
        return MetricsCollector(mock_redis, mock_session_factory, mock_priority_queue)

    async def test_record_job_completion_stores_sample(self, collector, mock_redis):
        """record_job_completion should store a sample in the sorted set."""
        await collector.record_job_completion(
            queue_time_ms=50.0, execution_time_ms=150.0
        )

        pipe = mock_redis.pipeline()
        pipe.zadd.assert_called_once()
        pipe.incr.assert_called_once()
        pipe.zremrangebyscore.assert_called_once()
        pipe.execute.assert_called_once()

    async def test_get_metrics_returns_zeros_when_no_samples(
        self, collector, mock_redis, mock_priority_queue
    ):
        """get_metrics should return zeros when no jobs have been processed."""
        mock_redis.zremrangebyscore = AsyncMock()
        mock_redis.zrangebyscore = AsyncMock(return_value=[])
        mock_redis.zcard = AsyncMock(return_value=0)
        mock_priority_queue.depth = AsyncMock(return_value=0)

        metrics = await collector.get_metrics()

        assert metrics["queue_depth"] == 0
        assert metrics["active_workers"] == 0
        assert metrics["jobs_per_second"] == 0.0
        assert metrics["latency_p50_ms"] == 0.0
        assert metrics["latency_p95_ms"] == 0.0
        assert metrics["dlq_size"] == 0

    async def test_get_metrics_calculates_throughput(
        self, collector, mock_redis, mock_priority_queue
    ):
        """get_metrics should calculate jobs_per_second from sample count."""
        # Simulate 30 samples in the last 60 seconds
        now_ms = time.time() * 1000
        samples = [
            json.dumps({"queue_time_ms": 10, "execution_time_ms": 20, "total_ms": 30, "timestamp": now_ms})
            for _ in range(30)
        ]

        mock_redis.zremrangebyscore = AsyncMock()
        mock_redis.zrangebyscore = AsyncMock(return_value=samples)
        mock_redis.zcard = AsyncMock(return_value=0)
        mock_priority_queue.depth = AsyncMock(return_value=5)

        metrics = await collector.get_metrics()

        assert metrics["jobs_per_second"] == 30 / 60.0
        assert metrics["queue_depth"] == 5

    async def test_get_metrics_calculates_percentiles(
        self, collector, mock_redis, mock_priority_queue
    ):
        """get_metrics should return correct P50 and P95 from samples."""
        now_ms = time.time() * 1000
        # Create 100 samples with total_ms from 1 to 100
        samples = [
            json.dumps({"queue_time_ms": i, "execution_time_ms": 0, "total_ms": float(i), "timestamp": now_ms})
            for i in range(1, 101)
        ]

        mock_redis.zremrangebyscore = AsyncMock()
        mock_redis.zrangebyscore = AsyncMock(return_value=samples)
        mock_redis.zcard = AsyncMock(return_value=2)
        mock_priority_queue.depth = AsyncMock(return_value=0)

        metrics = await collector.get_metrics()

        # P50 of 1..100: rank = 0.5 * 99 = 49.5 → interpolate between index 49 (50) and 50 (51)
        assert metrics["latency_p50_ms"] == 50.5
        # P95 of 1..100: rank = 0.95 * 99 = 94.05 → interpolate between index 94 (95) and 95 (96)
        assert 95.0 <= metrics["latency_p95_ms"] <= 96.0
        assert metrics["dlq_size"] == 2


@requires_redis
class TestMetricsCollectorIntegration:
    """Integration tests for MetricsCollector against a real Redis instance."""

    @pytest.fixture
    async def redis_client(self):
        """Create a Redis client for testing."""
        import redis.asyncio as aioredis

        client = aioredis.Redis(
            host="localhost",
            port=6379,
            db=2,  # Use DB 2 for metrics tests
            decode_responses=True,
        )
        yield client
        # Cleanup
        await client.delete(LATENCY_SAMPLES_KEY)
        await client.delete("metrics:completed_count")
        await client.delete(DLQ_KEY)
        await client.delete("queue:priority")
        await client.aclose()

    @pytest.fixture
    def mock_session_factory(self):
        """Create a mock session factory for integration tests."""
        session = AsyncMock()
        result = MagicMock()
        result.scalar_one.return_value = 2
        session.execute = AsyncMock(return_value=result)
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=None)

        factory = MagicMock()
        factory.return_value = session
        return factory

    @pytest.fixture
    def mock_priority_queue(self):
        """Create a mock priority queue for integration tests."""
        queue = AsyncMock()
        queue.depth = AsyncMock(return_value=3)
        return queue

    @pytest.fixture
    def collector(self, redis_client, mock_session_factory, mock_priority_queue):
        return MetricsCollector(redis_client, mock_session_factory, mock_priority_queue)

    async def test_record_and_retrieve_metrics(self, collector, redis_client):
        """Recording completions should be reflected in get_metrics."""
        # Record several job completions
        await collector.record_job_completion(queue_time_ms=10.0, execution_time_ms=90.0)
        await collector.record_job_completion(queue_time_ms=20.0, execution_time_ms=80.0)
        await collector.record_job_completion(queue_time_ms=15.0, execution_time_ms=85.0)

        metrics = await collector.get_metrics()

        assert metrics["jobs_per_second"] == 3 / 60.0
        assert metrics["queue_depth"] == 3
        assert metrics["active_workers"] == 2
        # All total_ms values: 100, 100, 100 - all the same
        assert metrics["latency_p50_ms"] == 100.0
        assert metrics["latency_p95_ms"] == 100.0

    async def test_returns_zeros_with_no_completions(self, collector, redis_client):
        """Should return zeros when no jobs have been processed."""
        # Override queue depth to 0 for this test
        collector._priority_queue.depth = AsyncMock(return_value=0)

        metrics = await collector.get_metrics()

        assert metrics["jobs_per_second"] == 0.0
        assert metrics["latency_p50_ms"] == 0.0
        assert metrics["latency_p95_ms"] == 0.0

    async def test_sliding_window_expiry(self, collector, redis_client):
        """Samples older than 60 seconds should be excluded from metrics."""
        # Insert an old sample directly (timestamp 70 seconds ago)
        old_timestamp_ms = (time.time() - 70) * 1000
        old_sample = json.dumps({
            "queue_time_ms": 500.0,
            "execution_time_ms": 500.0,
            "total_ms": 1000.0,
            "timestamp": old_timestamp_ms,
        })
        await redis_client.zadd(LATENCY_SAMPLES_KEY, {old_sample: old_timestamp_ms})

        # Record a fresh sample
        await collector.record_job_completion(queue_time_ms=10.0, execution_time_ms=40.0)

        metrics = await collector.get_metrics()

        # Only the fresh sample should be counted (old one expired)
        assert metrics["jobs_per_second"] == 1 / 60.0
        assert metrics["latency_p50_ms"] == 50.0

    async def test_percentile_with_varied_latencies(self, collector, redis_client):
        """Percentiles should reflect the distribution of latencies."""
        # Record 10 fast jobs and 1 slow job
        for _ in range(10):
            await collector.record_job_completion(queue_time_ms=5.0, execution_time_ms=15.0)

        await collector.record_job_completion(queue_time_ms=100.0, execution_time_ms=400.0)

        metrics = await collector.get_metrics()

        # P50 should be near 20ms (the fast jobs)
        assert metrics["latency_p50_ms"] == 20.0
        # P95 should be much higher (influenced by the slow job)
        assert metrics["latency_p95_ms"] > 20.0

    async def test_dlq_size_reported(self, collector, redis_client):
        """DLQ size should reflect the number of items in queue:dlq."""
        # Add items to DLQ sorted set
        await redis_client.zadd(DLQ_KEY, {"job1": 1.0, "job2": 2.0})

        metrics = await collector.get_metrics()

        assert metrics["dlq_size"] == 2
