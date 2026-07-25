"""Property-based tests for distributed lock mutual exclusion.

**Validates: Requirements 11.1, 11.3, 15.5**

Uses Hypothesis to verify:
- Property 7: No Double Execution (Lock Mutual Exclusion) — at most one worker
  holds a lock for a given job at any time (Req 11.1, 15.5)
- Property 8: Lock Owner-Only Release — only the lock holder can release
  the lock (Req 11.3)
"""

import asyncio
from uuid import UUID, uuid4

import fakeredis.aioredis
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from src.core.distributed_lock import DistributedLock


# --- Strategies ---

uuid_strategy = st.uuids()

# TTL between 31 and 600 seconds (timeout_seconds + 30 buffer minimum)
ttl_strategy = st.integers(min_value=31, max_value=600)

# Generate lists of distinct worker UUIDs (2-10 workers competing)
worker_list_strategy = st.lists(
    uuid_strategy, min_size=2, max_size=10, unique=True
)


# --- Fixtures ---

@pytest.fixture
def fake_redis():
    """Create a fakeredis async client with Lua scripting support."""
    return fakeredis.aioredis.FakeRedis(decode_responses=True)


@pytest.fixture
def lock(fake_redis):
    """Create a DistributedLock instance backed by fakeredis."""
    return DistributedLock(redis_client=fake_redis)


# --- Property Tests ---


class TestNoDoubleExecution:
    """Property 7: No Double Execution (Lock Mutual Exclusion).

    At most one worker holds a lock for a given job at any time.
    When multiple workers attempt to acquire the same lock concurrently,
    exactly one succeeds and the rest fail.

    **Validates: Requirements 11.1, 15.5**
    """

    @given(job_id=uuid_strategy, workers=worker_list_strategy, ttl=ttl_strategy)
    @settings(max_examples=200)
    async def test_exactly_one_worker_acquires_lock(
        self, job_id: UUID, workers: list[UUID], ttl: int
    ):
        """When N workers try to acquire the same lock, exactly one succeeds.

        **Validates: Requirements 11.1, 15.5**
        """
        redis_client = fakeredis.aioredis.FakeRedis(decode_responses=True)
        lock = DistributedLock(redis_client=redis_client)

        results = []
        for worker_id in workers:
            acquired = await lock.acquire_lock(job_id, worker_id, ttl)
            results.append((worker_id, acquired))

        # Exactly one worker should have acquired the lock
        successful = [w for w, acquired in results if acquired]
        assert len(successful) == 1

        # The lock value in Redis should be the successful worker's ID
        holder = await lock.get_lock_holder(job_id)
        assert holder == str(successful[0])

        await redis_client.aclose()

    @given(
        job_id=uuid_strategy,
        worker_a=uuid_strategy,
        worker_b=uuid_strategy,
        ttl=ttl_strategy,
    )
    @settings(max_examples=200)
    async def test_second_acquire_fails_while_lock_held(
        self, job_id: UUID, worker_a: UUID, worker_b: UUID, ttl: int
    ):
        """A lock cannot be acquired by a second worker while held.

        **Validates: Requirements 11.1**
        """
        assume(worker_a != worker_b)

        redis_client = fakeredis.aioredis.FakeRedis(decode_responses=True)
        lock = DistributedLock(redis_client=redis_client)

        # Worker A acquires
        assert await lock.acquire_lock(job_id, worker_a, ttl) is True
        # Worker B cannot acquire
        assert await lock.acquire_lock(job_id, worker_b, ttl) is False
        # Lock is still held by Worker A
        assert await lock.get_lock_holder(job_id) == str(worker_a)

        await redis_client.aclose()

    @given(
        job_id=uuid_strategy,
        worker_id=uuid_strategy,
        ttl=ttl_strategy,
    )
    @settings(max_examples=200)
    async def test_same_worker_cannot_double_acquire(
        self, job_id: UUID, worker_id: UUID, ttl: int
    ):
        """Even the same worker cannot acquire a lock it already holds (NX semantics).

        **Validates: Requirements 11.1**
        """
        redis_client = fakeredis.aioredis.FakeRedis(decode_responses=True)
        lock = DistributedLock(redis_client=redis_client)

        first = await lock.acquire_lock(job_id, worker_id, ttl)
        second = await lock.acquire_lock(job_id, worker_id, ttl)

        assert first is True
        assert second is False

        await redis_client.aclose()

    @given(
        data=st.data(),
        workers=worker_list_strategy,
        ttl=ttl_strategy,
    )
    @settings(max_examples=200)
    async def test_different_jobs_can_be_locked_independently(
        self, data, workers: list[UUID], ttl: int
    ):
        """Different jobs can each have their own lock held by different workers.

        **Validates: Requirements 11.1**
        """
        # Generate as many distinct job IDs as workers
        job_ids = [data.draw(uuid_strategy) for _ in workers]
        # Ensure job IDs are unique
        assume(len(set(job_ids)) == len(job_ids))

        redis_client = fakeredis.aioredis.FakeRedis(decode_responses=True)
        lock = DistributedLock(redis_client=redis_client)

        # Each worker acquires a lock on a different job
        for job_id, worker_id in zip(job_ids, workers):
            acquired = await lock.acquire_lock(job_id, worker_id, ttl)
            assert acquired is True

        # Each job is locked by its corresponding worker
        for job_id, worker_id in zip(job_ids, workers):
            holder = await lock.get_lock_holder(job_id)
            assert holder == str(worker_id)

        await redis_client.aclose()


class TestLockOwnerOnlyRelease:
    """Property 8: Lock Owner-Only Release.

    Only the lock holder (matching worker_id) can release the lock.
    Non-holders attempting to release have no effect.

    **Validates: Requirements 11.3**
    """

    @given(
        job_id=uuid_strategy,
        holder=uuid_strategy,
        non_holder=uuid_strategy,
        ttl=ttl_strategy,
    )
    @settings(max_examples=200)
    async def test_only_holder_can_release(
        self, job_id: UUID, holder: UUID, non_holder: UUID, ttl: int
    ):
        """A non-holder's release attempt returns False and leaves lock intact.

        **Validates: Requirements 11.3**
        """
        assume(holder != non_holder)

        redis_client = fakeredis.aioredis.FakeRedis(decode_responses=True)
        lock = DistributedLock(redis_client=redis_client)

        # Holder acquires the lock
        await lock.acquire_lock(job_id, holder, ttl)

        # Non-holder tries to release — must fail
        released = await lock.release_lock(job_id, non_holder)
        assert released is False

        # Lock is still held by the original holder
        assert await lock.is_locked(job_id) is True
        assert await lock.get_lock_holder(job_id) == str(holder)

        await redis_client.aclose()

    @given(
        job_id=uuid_strategy,
        holder=uuid_strategy,
        ttl=ttl_strategy,
    )
    @settings(max_examples=200)
    async def test_holder_can_release_own_lock(
        self, job_id: UUID, holder: UUID, ttl: int
    ):
        """The lock holder can release the lock successfully.

        **Validates: Requirements 11.3**
        """
        redis_client = fakeredis.aioredis.FakeRedis(decode_responses=True)
        lock = DistributedLock(redis_client=redis_client)

        await lock.acquire_lock(job_id, holder, ttl)
        released = await lock.release_lock(job_id, holder)

        assert released is True
        assert await lock.is_locked(job_id) is False

        await redis_client.aclose()

    @given(
        job_id=uuid_strategy,
        holder=uuid_strategy,
        non_holders=st.lists(uuid_strategy, min_size=1, max_size=5),
        ttl=ttl_strategy,
    )
    @settings(max_examples=200)
    async def test_multiple_non_holders_cannot_release(
        self, job_id: UUID, holder: UUID, non_holders: list[UUID], ttl: int
    ):
        """Multiple different non-holders all fail to release the lock.

        **Validates: Requirements 11.3**
        """
        # Ensure no non-holder is the same as holder
        non_holders = [w for w in non_holders if w != holder]
        assume(len(non_holders) >= 1)

        redis_client = fakeredis.aioredis.FakeRedis(decode_responses=True)
        lock = DistributedLock(redis_client=redis_client)

        await lock.acquire_lock(job_id, holder, ttl)

        # Every non-holder fails to release
        for non_holder in non_holders:
            released = await lock.release_lock(job_id, non_holder)
            assert released is False

        # Lock remains held by original holder
        assert await lock.get_lock_holder(job_id) == str(holder)

        await redis_client.aclose()

    @given(
        job_id=uuid_strategy,
        worker_a=uuid_strategy,
        worker_b=uuid_strategy,
        ttl=ttl_strategy,
    )
    @settings(max_examples=200)
    async def test_release_after_reacquisition_by_another_worker(
        self, job_id: UUID, worker_a: UUID, worker_b: UUID, ttl: int
    ):
        """Original holder cannot release after lock is released and re-acquired by another.

        **Validates: Requirements 11.3**
        """
        assume(worker_a != worker_b)

        redis_client = fakeredis.aioredis.FakeRedis(decode_responses=True)
        lock = DistributedLock(redis_client=redis_client)

        # Worker A acquires and releases
        await lock.acquire_lock(job_id, worker_a, ttl)
        await lock.release_lock(job_id, worker_a)

        # Worker B acquires
        acquired = await lock.acquire_lock(job_id, worker_b, ttl)
        assert acquired is True

        # Worker A cannot release Worker B's lock
        released_by_a = await lock.release_lock(job_id, worker_a)
        assert released_by_a is False

        # Worker B still holds the lock
        assert await lock.get_lock_holder(job_id) == str(worker_b)

        await redis_client.aclose()

    @given(
        job_id=uuid_strategy,
        worker_id=uuid_strategy,
    )
    @settings(max_examples=200)
    async def test_release_nonexistent_lock_returns_false(
        self, job_id: UUID, worker_id: UUID
    ):
        """Releasing a lock that was never acquired returns False.

        **Validates: Requirements 11.3**
        """
        redis_client = fakeredis.aioredis.FakeRedis(decode_responses=True)
        lock = DistributedLock(redis_client=redis_client)

        released = await lock.release_lock(job_id, worker_id)
        assert released is False

        await redis_client.aclose()
