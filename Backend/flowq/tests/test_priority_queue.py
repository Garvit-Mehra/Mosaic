"""Unit tests for the Redis priority queue module.

Tests cover the score calculation logic (pure function) and
integration tests for the queue operations against a real Redis instance.
"""

import pytest
from datetime import datetime, timezone
from uuid import uuid4

from src.core.priority_queue import calculate_queue_score, RedisPriorityQueue


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


class TestCalculateQueueScore:
    """Tests for the pure score calculation function."""

    def test_higher_priority_produces_lower_score(self):
        """Higher priority value should result in a lower (more negative) score."""
        t = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        score_low = calculate_queue_score(1, t)
        score_high = calculate_queue_score(10, t)
        assert score_high < score_low

    def test_same_priority_earlier_time_produces_lower_score(self):
        """With equal priority, earlier enqueue time should produce lower score (FIFO)."""
        t_early = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        t_late = datetime(2024, 1, 1, 12, 0, 1, tzinfo=timezone.utc)
        score_early = calculate_queue_score(5, t_early)
        score_late = calculate_queue_score(5, t_late)
        assert score_early < score_late

    def test_score_formula_exact(self):
        """Verify the exact score formula: -priority * 1e12 + timestamp_ms."""
        t = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        timestamp_ms = int(t.timestamp() * 1000)
        priority = 3
        expected = -priority * 1_000_000_000_000 + timestamp_ms
        result = calculate_queue_score(priority, t)
        assert result == float(expected)

    def test_zero_priority(self):
        """Priority 0 should produce score equal to timestamp_ms."""
        t = datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc)
        timestamp_ms = int(t.timestamp() * 1000)
        score = calculate_queue_score(0, t)
        assert score == float(timestamp_ms)

    def test_score_differentiates_millisecond_timestamps(self):
        """Jobs enqueued 1ms apart with same priority should have different scores."""
        from datetime import timedelta

        t1 = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        t2 = t1 + timedelta(milliseconds=1)
        score1 = calculate_queue_score(5, t1)
        score2 = calculate_queue_score(5, t2)
        assert score1 != score2
        assert score1 < score2


@requires_redis
class TestRedisPriorityQueueIntegration:
    """Integration tests for RedisPriorityQueue against a real Redis instance.

    These tests require a running Redis server on localhost:6379.
    They use a unique queue key per test to avoid interference.
    """

    @pytest.fixture
    async def queue(self):
        """Create a queue with a unique test key and clean up after."""
        import redis.asyncio as redis

        client = redis.Redis(
            host="localhost",
            port=6379,
            db=1,  # Use DB 1 for testing
            decode_responses=True,
        )
        test_key = f"test:queue:{uuid4().hex[:8]}"
        q = RedisPriorityQueue(redis_client=client, queue_key=test_key)
        yield q
        # Cleanup
        await client.delete(test_key)
        await client.aclose()

    async def test_enqueue_and_depth(self, queue):
        """Enqueuing jobs increases the queue depth."""
        assert await queue.depth() == 0
        job_id = uuid4()
        await queue.enqueue(job_id, priority=5, enqueued_at=datetime.now(timezone.utc))
        assert await queue.depth() == 1

    async def test_enqueue_multiple_and_depth(self, queue):
        """Queue depth reflects all enqueued jobs."""
        now = datetime.now(timezone.utc)
        for _ in range(5):
            await queue.enqueue(uuid4(), priority=1, enqueued_at=now)
        assert await queue.depth() == 5

    async def test_dequeue_returns_highest_priority_first(self, queue):
        """Dequeue should return the highest-priority job first."""
        from datetime import timedelta

        now = datetime.now(timezone.utc)
        low_id = uuid4()
        high_id = uuid4()

        await queue.enqueue(low_id, priority=1, enqueued_at=now)
        await queue.enqueue(high_id, priority=10, enqueued_at=now + timedelta(seconds=1))

        result = await queue.dequeue(timeout=1.0)
        assert result == high_id

    async def test_dequeue_fifo_for_same_priority(self, queue):
        """Jobs with same priority should dequeue in FIFO order."""
        from datetime import timedelta

        now = datetime.now(timezone.utc)
        first_id = uuid4()
        second_id = uuid4()

        await queue.enqueue(first_id, priority=5, enqueued_at=now)
        await queue.enqueue(second_id, priority=5, enqueued_at=now + timedelta(milliseconds=100))

        result1 = await queue.dequeue(timeout=1.0)
        result2 = await queue.dequeue(timeout=1.0)
        assert result1 == first_id
        assert result2 == second_id

    async def test_dequeue_returns_none_on_empty_queue(self, queue):
        """Dequeue on empty queue should return None after timeout."""
        result = await queue.dequeue(timeout=0.5)
        assert result is None

    async def test_remove_existing_job(self, queue):
        """Removing an existing job should return True and reduce depth."""
        job_id = uuid4()
        await queue.enqueue(job_id, priority=3, enqueued_at=datetime.now(timezone.utc))
        assert await queue.depth() == 1

        removed = await queue.remove(job_id)
        assert removed is True
        assert await queue.depth() == 0

    async def test_remove_nonexistent_job(self, queue):
        """Removing a job not in the queue should return False."""
        removed = await queue.remove(uuid4())
        assert removed is False

    async def test_peek_returns_jobs_in_priority_order(self, queue):
        """Peek should return jobs ordered by priority (highest first)."""
        from datetime import timedelta

        now = datetime.now(timezone.utc)
        low_id = uuid4()
        mid_id = uuid4()
        high_id = uuid4()

        await queue.enqueue(low_id, priority=1, enqueued_at=now)
        await queue.enqueue(mid_id, priority=5, enqueued_at=now + timedelta(seconds=1))
        await queue.enqueue(high_id, priority=10, enqueued_at=now + timedelta(seconds=2))

        peeked = await queue.peek(count=3)
        assert peeked == [high_id, mid_id, low_id]

    async def test_peek_does_not_remove_jobs(self, queue):
        """Peek should not modify the queue."""
        job_id = uuid4()
        await queue.enqueue(job_id, priority=5, enqueued_at=datetime.now(timezone.utc))

        await queue.peek(count=5)
        assert await queue.depth() == 1

    async def test_peek_with_count_less_than_depth(self, queue):
        """Peek with count < depth returns only the requested number."""
        now = datetime.now(timezone.utc)
        for i in range(5):
            await queue.enqueue(uuid4(), priority=i, enqueued_at=now)

        peeked = await queue.peek(count=2)
        assert len(peeked) == 2

    async def test_dequeue_removes_from_queue(self, queue):
        """Dequeuing a job should remove it from the queue."""
        job_id = uuid4()
        await queue.enqueue(job_id, priority=5, enqueued_at=datetime.now(timezone.utc))
        assert await queue.depth() == 1

        await queue.dequeue(timeout=1.0)
        assert await queue.depth() == 0
