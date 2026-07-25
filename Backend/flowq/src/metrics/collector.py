"""Metrics collector for tracking job queue performance.

Tracks latency components (queue time and execution time), jobs-per-second
throughput, and provides percentile calculations over a 60-second sliding window.

Requirements: 14.1, 14.3, 14.5
"""

import json
import time
from typing import Optional

import redis.asyncio as redis
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.core.priority_queue import PriorityQueueInterface
from src.models.enums import WorkerStatus
from src.models.worker import Worker

# Redis keys used by the metrics collector
LATENCY_SAMPLES_KEY = "metrics:latency_samples"
COMPLETED_COUNT_KEY = "metrics:completed_count"
DLQ_KEY = "queue:dlq"

# Sliding window duration in milliseconds
WINDOW_MS = 60_000


class MetricsCollector:
    """Collects and reports job queue metrics.

    Uses a Redis sorted set for latency samples with timestamps as scores,
    enabling efficient time-based windowing and cleanup.
    """

    def __init__(
        self,
        redis_client: redis.Redis,
        session_factory: async_sessionmaker[AsyncSession],
        priority_queue: PriorityQueueInterface,
    ):
        """Initialize the metrics collector.

        Args:
            redis_client: Async Redis client for storing latency samples.
            session_factory: SQLAlchemy async session factory for querying workers.
            priority_queue: Priority queue interface for reading queue depth.
        """
        self._redis = redis_client
        self._session_factory = session_factory
        self._priority_queue = priority_queue

    async def record_job_completion(
        self, queue_time_ms: float, execution_time_ms: float
    ) -> None:
        """Record a latency sample when a job completes.

        Stores a sample in the Redis sorted set with the current timestamp as
        the score, and increments the completion counter.

        Args:
            queue_time_ms: Time spent in queue (enqueue to dequeue) in milliseconds.
            execution_time_ms: Time spent executing (start to completion) in milliseconds.
        """
        now_ms = _current_time_ms()
        total_ms = queue_time_ms + execution_time_ms

        sample = json.dumps({
            "queue_time_ms": queue_time_ms,
            "execution_time_ms": execution_time_ms,
            "total_ms": total_ms,
            "timestamp": now_ms,
        })

        pipe = self._redis.pipeline()
        pipe.zadd(LATENCY_SAMPLES_KEY, {sample: now_ms})
        pipe.incr(COMPLETED_COUNT_KEY)
        # Clean up expired samples older than the sliding window
        pipe.zremrangebyscore(LATENCY_SAMPLES_KEY, 0, now_ms - WINDOW_MS)
        await pipe.execute()

    async def get_metrics(self) -> dict:
        """Get current metrics snapshot.

        Returns a dictionary containing:
            - queue_depth: Number of jobs in the priority queue
            - active_workers: Number of workers with ACTIVE status
            - jobs_per_second: Throughput over the last 60 seconds
            - latency_p50_ms: 50th percentile total latency
            - latency_p95_ms: 95th percentile total latency
            - dlq_size: Number of jobs in the dead-letter queue

        Returns:
            Dict with metrics values. Returns zeros when no jobs have been processed.
        """
        now_ms = _current_time_ms()

        # Clean up expired samples
        await self._redis.zremrangebyscore(LATENCY_SAMPLES_KEY, 0, now_ms - WINDOW_MS)

        # Gather data in parallel where possible
        queue_depth = await self._priority_queue.depth()
        active_workers = await self._get_active_worker_count()
        dlq_size = await self._redis.zcard(DLQ_KEY)

        # Get latency samples from the last 60 seconds
        raw_samples = await self._redis.zrangebyscore(
            LATENCY_SAMPLES_KEY, now_ms - WINDOW_MS, now_ms
        )

        if not raw_samples:
            return {
                "queue_depth": queue_depth,
                "active_workers": active_workers,
                "jobs_per_second": 0.0,
                "latency_p50_ms": 0.0,
                "latency_p95_ms": 0.0,
                "dlq_size": dlq_size,
            }

        # Parse samples and calculate metrics
        total_ms_values = []
        for raw in raw_samples:
            sample = json.loads(raw)
            total_ms_values.append(sample["total_ms"])

        jobs_per_second = len(total_ms_values) / 60.0
        total_ms_values.sort()
        p50 = _percentile(total_ms_values, 50)
        p95 = _percentile(total_ms_values, 95)

        return {
            "queue_depth": queue_depth,
            "active_workers": active_workers,
            "jobs_per_second": jobs_per_second,
            "latency_p50_ms": p50,
            "latency_p95_ms": p95,
            "dlq_size": dlq_size,
        }

    async def _get_active_worker_count(self) -> int:
        """Query PostgreSQL for the count of active workers."""
        async with self._session_factory() as session:
            result = await session.execute(
                select(func.count(Worker.id)).where(
                    Worker.status == WorkerStatus.ACTIVE
                )
            )
            return result.scalar_one()


def _current_time_ms() -> float:
    """Get current time in milliseconds since epoch."""
    return time.time() * 1000


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Calculate percentile from a sorted list of values.

    Uses the nearest-rank method for percentile calculation.

    Args:
        sorted_values: Pre-sorted list of numeric values.
        pct: Percentile to compute (0-100).

    Returns:
        The value at the given percentile. Returns 0.0 for empty lists.
    """
    if not sorted_values:
        return 0.0

    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]

    # Nearest-rank method
    rank = (pct / 100.0) * (n - 1)
    lower = int(rank)
    upper = min(lower + 1, n - 1)
    fraction = rank - lower

    # Linear interpolation between the two nearest ranks
    return sorted_values[lower] + fraction * (sorted_values[upper] - sorted_values[lower])
