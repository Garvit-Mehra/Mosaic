"""Redis-backed priority queue using sorted sets.

Implements priority ordering via score calculation:
score = -priority × 1_000_000_000_000 + enqueued_at_timestamp_ms

Higher priority produces lower scores, so ZPOPMIN returns highest-priority jobs first.
Jobs with equal priority are ordered FIFO by enqueue time.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional
from uuid import UUID

import redis.asyncio as redis

from src.redis_client import get_redis_client

QUEUE_KEY = "queue:priority"


def calculate_queue_score(priority: int, enqueued_at: datetime) -> float:
    """Calculate Redis sorted set score for priority ordering.

    Score formula: -priority * 1_000_000_000_000 + enqueued_at_timestamp_ms

    This ensures:
    - Higher priority (larger value) produces a lower score → popped first by ZPOPMIN.
    - Equal priority jobs are ordered by enqueue time (FIFO).

    Args:
        priority: Job priority level (>= 0, higher = more urgent).
        enqueued_at: Timestamp when the job was enqueued.

    Returns:
        The computed score as a float.
    """
    timestamp_ms = int(enqueued_at.timestamp() * 1000)
    score = -priority * 1_000_000_000_000 + timestamp_ms
    return float(score)


class PriorityQueueInterface(ABC):
    """Abstract interface for priority queue operations."""

    @abstractmethod
    async def enqueue(self, job_id: UUID, priority: int, enqueued_at: datetime) -> None:
        """Add job to priority queue. Score = -priority * 1e12 + timestamp_ms."""
        ...

    @abstractmethod
    async def dequeue(self, timeout: float = 5.0) -> Optional[UUID]:
        """Blocking pop of highest-priority job. Returns None on timeout."""
        ...

    @abstractmethod
    async def remove(self, job_id: UUID) -> bool:
        """Remove job from queue (for cancellation). Returns False if not found."""
        ...

    @abstractmethod
    async def depth(self) -> int:
        """Current number of jobs in queue."""
        ...

    @abstractmethod
    async def peek(self, count: int = 10) -> List[UUID]:
        """Preview top N jobs without removing them."""
        ...


class RedisPriorityQueue(PriorityQueueInterface):
    """Redis sorted set-backed priority queue implementation.

    Uses ZADD, BZPOPMIN, ZREM, ZCARD, and ZRANGE for atomic queue operations.
    """

    def __init__(self, redis_client: Optional[redis.Redis] = None, queue_key: str = QUEUE_KEY):
        """Initialize the priority queue.

        Args:
            redis_client: Optional Redis client instance. If None, uses the shared client.
            queue_key: The Redis key for the sorted set. Defaults to "queue:priority".
        """
        self._redis = redis_client
        self._queue_key = queue_key

    @property
    def redis(self) -> redis.Redis:
        """Get the Redis client, falling back to the shared instance."""
        if self._redis is None:
            self._redis = get_redis_client()
        return self._redis

    async def enqueue(self, job_id: UUID, priority: int, enqueued_at: datetime) -> None:
        """Add a job to the priority queue using ZADD.

        Args:
            job_id: Unique identifier of the job.
            priority: Job priority (higher = more urgent).
            enqueued_at: Timestamp when the job was enqueued.
        """
        score = calculate_queue_score(priority, enqueued_at)
        await self.redis.zadd(self._queue_key, {str(job_id): score})

    async def dequeue(self, timeout: float = 5.0) -> Optional[UUID]:
        """Blocking pop of the highest-priority job using BZPOPMIN.

        Blocks for up to `timeout` seconds waiting for a job. Returns None
        if no job becomes available within the timeout period.

        Args:
            timeout: Maximum seconds to block waiting for a job.

        Returns:
            The UUID of the dequeued job, or None if timeout elapsed.
        """
        result = await self.redis.bzpopmin(self._queue_key, timeout=timeout)
        if result is None:
            return None
        # BZPOPMIN returns (key, member, score) when decode_responses=True
        # or (key, (member, score)) depending on redis-py version
        # With decode_responses=True: returns (key, member, score) as a tuple
        _key, member, _score = result
        return UUID(member)

    async def remove(self, job_id: UUID) -> bool:
        """Remove a job from the queue using ZREM (for cancellation).

        Args:
            job_id: The job to remove from the queue.

        Returns:
            True if the job was found and removed, False if not in the queue.
        """
        removed = await self.redis.zrem(self._queue_key, str(job_id))
        return removed > 0

    async def depth(self) -> int:
        """Get the current number of jobs in the queue using ZCARD.

        Returns:
            The number of jobs currently in the priority queue.
        """
        return await self.redis.zcard(self._queue_key)

    async def peek(self, count: int = 10) -> List[UUID]:
        """Preview the top N jobs without removing them using ZRANGE.

        Returns jobs ordered by priority (highest priority first, i.e., lowest score).

        Args:
            count: Number of jobs to preview (default 10).

        Returns:
            List of job UUIDs in priority order.
        """
        # ZRANGE with start=0, end=count-1 returns members with lowest scores first
        members = await self.redis.zrange(self._queue_key, 0, count - 1)
        return [UUID(member) for member in members]
