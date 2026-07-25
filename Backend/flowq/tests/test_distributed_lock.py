"""Tests for distributed lock acquisition and release.

Unit tests for lock logic and integration tests requiring Redis.
"""

import asyncio
from uuid import uuid4

import pytest
import redis.asyncio as redis

from src.core.distributed_lock import (
    LOCK_TTL_BUFFER,
    RELEASE_LOCK_SCRIPT,
    DistributedLock,
    _lock_key,
    acquire_lock,
    release_lock,
)


@pytest.fixture
async def redis_client():
    """Create a real Redis client for integration tests."""
    client = redis.Redis(host="localhost", port=6379, db=1, decode_responses=True)
    try:
        await client.ping()
    except redis.ConnectionError:
        pytest.skip("Redis not available for integration tests")
    yield client
    # Clean up any leftover lock keys
    keys = await client.keys("lock:job:*")
    if keys:
        await client.delete(*keys)
    await client.aclose()


@pytest.fixture
def lock(redis_client):
    """Create a DistributedLock instance with test Redis client."""
    return DistributedLock(redis_client=redis_client)


class TestLockKeyFormat:
    """Test lock key formatting."""

    def test_lock_key_format(self):
        """Lock key should follow format lock:job:{job_id}."""
        job_id = uuid4()
        key = _lock_key(job_id)
        assert key == f"lock:job:{job_id}"

    def test_lock_key_uses_uuid_string(self):
        """Lock key should use UUID string representation."""
        job_id = uuid4()
        key = _lock_key(job_id)
        assert str(job_id) in key


class TestAcquireLock:
    """Test lock acquisition logic (Requirement 11.1, 11.2)."""

    async def test_acquire_lock_success(self, lock, redis_client):
        """Acquiring a lock for an unlocked job should succeed."""
        job_id = uuid4()
        worker_id = uuid4()
        ttl = 330  # 300 timeout + 30 buffer

        result = await lock.acquire_lock(job_id, worker_id, ttl)

        assert result is True
        # Verify key was set in Redis
        stored = await redis_client.get(_lock_key(job_id))
        assert stored == str(worker_id)

    async def test_acquire_lock_sets_ttl(self, lock, redis_client):
        """Lock should have TTL set (Requirement 11.2)."""
        job_id = uuid4()
        worker_id = uuid4()
        ttl = 330

        await lock.acquire_lock(job_id, worker_id, ttl)

        remaining = await redis_client.ttl(_lock_key(job_id))
        # TTL should be close to what we set (within a second of execution time)
        assert 0 < remaining <= ttl

    async def test_acquire_lock_mutual_exclusion(self, lock, redis_client):
        """Only one worker should hold the lock at a time (Requirement 11.1)."""
        job_id = uuid4()
        worker_a = uuid4()
        worker_b = uuid4()
        ttl = 330

        # Worker A acquires
        result_a = await lock.acquire_lock(job_id, worker_a, ttl)
        # Worker B tries to acquire same job
        result_b = await lock.acquire_lock(job_id, worker_b, ttl)

        assert result_a is True
        assert result_b is False
        # Verify Worker A still holds the lock
        stored = await redis_client.get(_lock_key(job_id))
        assert stored == str(worker_a)

    async def test_acquire_lock_different_jobs(self, lock, redis_client):
        """Different jobs can be locked by different workers simultaneously."""
        job_1 = uuid4()
        job_2 = uuid4()
        worker_a = uuid4()
        worker_b = uuid4()
        ttl = 330

        result_1 = await lock.acquire_lock(job_1, worker_a, ttl)
        result_2 = await lock.acquire_lock(job_2, worker_b, ttl)

        assert result_1 is True
        assert result_2 is True

    async def test_acquire_lock_same_worker_same_job(self, lock, redis_client):
        """Same worker cannot double-acquire the same lock (NX semantics)."""
        job_id = uuid4()
        worker_id = uuid4()
        ttl = 330

        first = await lock.acquire_lock(job_id, worker_id, ttl)
        second = await lock.acquire_lock(job_id, worker_id, ttl)

        assert first is True
        assert second is False


class TestReleaseLock:
    """Test lock release logic (Requirement 11.3, 11.5)."""

    async def test_release_lock_by_holder(self, lock, redis_client):
        """Lock holder should be able to release the lock."""
        job_id = uuid4()
        worker_id = uuid4()
        ttl = 330

        await lock.acquire_lock(job_id, worker_id, ttl)
        result = await lock.release_lock(job_id, worker_id)

        assert result is True
        # Verify key is deleted
        exists = await redis_client.exists(_lock_key(job_id))
        assert exists == 0

    async def test_release_lock_by_non_holder(self, lock, redis_client):
        """Non-holder should not be able to release the lock (Requirement 11.3)."""
        job_id = uuid4()
        worker_a = uuid4()
        worker_b = uuid4()
        ttl = 330

        await lock.acquire_lock(job_id, worker_a, ttl)
        result = await lock.release_lock(job_id, worker_b)

        assert result is False
        # Verify lock still held by worker A
        stored = await redis_client.get(_lock_key(job_id))
        assert stored == str(worker_a)

    async def test_release_lock_not_held(self, lock, redis_client):
        """Releasing a lock that doesn't exist should return False."""
        job_id = uuid4()
        worker_id = uuid4()

        result = await lock.release_lock(job_id, worker_id)

        assert result is False

    async def test_release_lock_expired(self, lock, redis_client):
        """Releasing an expired lock should return False (Requirement 11.5).

        When release_lock returns False, the caller should discard its result.
        """
        job_id = uuid4()
        worker_id = uuid4()
        ttl = 1  # 1 second TTL

        await lock.acquire_lock(job_id, worker_id, ttl)
        # Wait for TTL to expire
        await asyncio.sleep(1.5)

        result = await lock.release_lock(job_id, worker_id)

        assert result is False

    async def test_release_lock_after_reacquisition(self, lock, redis_client):
        """Original holder cannot release lock after another worker acquires it.

        This tests the scenario where lock TTL expires and another worker
        re-acquires. The original worker's release should fail (Requirement 11.5).
        """
        job_id = uuid4()
        worker_a = uuid4()
        worker_b = uuid4()

        # Worker A acquires with short TTL
        await lock.acquire_lock(job_id, worker_a, ttl=1)
        # Wait for TTL to expire
        await asyncio.sleep(1.5)
        # Worker B acquires the now-available lock
        result_b = await lock.acquire_lock(job_id, worker_b, ttl=300)
        assert result_b is True

        # Worker A tries to release — should fail (lock is held by B now)
        result_release = await lock.release_lock(job_id, worker_a)
        assert result_release is False

        # Verify Worker B still holds the lock
        stored = await redis_client.get(_lock_key(job_id))
        assert stored == str(worker_b)


class TestLockAutoExpiry:
    """Test lock auto-expiry (Requirement 11.4)."""

    async def test_lock_auto_expires(self, lock, redis_client):
        """Lock should auto-expire after TTL, preventing deadlock."""
        job_id = uuid4()
        worker_id = uuid4()
        ttl = 1  # 1 second for fast test

        await lock.acquire_lock(job_id, worker_id, ttl)
        # Verify lock exists
        assert await redis_client.exists(_lock_key(job_id)) == 1

        # Wait for expiry
        await asyncio.sleep(1.5)

        # Lock should be gone
        assert await redis_client.exists(_lock_key(job_id)) == 0

    async def test_lock_reacquirable_after_expiry(self, lock, redis_client):
        """Another worker can acquire the lock after TTL expiry (Requirement 11.4)."""
        job_id = uuid4()
        worker_a = uuid4()
        worker_b = uuid4()

        # Worker A acquires with short TTL
        await lock.acquire_lock(job_id, worker_a, ttl=1)
        # Wait for expiry
        await asyncio.sleep(1.5)

        # Worker B should now be able to acquire
        result = await lock.acquire_lock(job_id, worker_b, ttl=300)
        assert result is True
        stored = await redis_client.get(_lock_key(job_id))
        assert stored == str(worker_b)


class TestHelperMethods:
    """Test is_locked and get_lock_holder helper methods."""

    async def test_is_locked_true(self, lock, redis_client):
        """is_locked should return True when lock is held."""
        job_id = uuid4()
        worker_id = uuid4()

        await lock.acquire_lock(job_id, worker_id, ttl=300)
        assert await lock.is_locked(job_id) is True

    async def test_is_locked_false(self, lock, redis_client):
        """is_locked should return False when no lock exists."""
        job_id = uuid4()
        assert await lock.is_locked(job_id) is False

    async def test_get_lock_holder(self, lock, redis_client):
        """get_lock_holder should return the holder's worker_id."""
        job_id = uuid4()
        worker_id = uuid4()

        await lock.acquire_lock(job_id, worker_id, ttl=300)
        holder = await lock.get_lock_holder(job_id)
        assert holder == str(worker_id)

    async def test_get_lock_holder_none(self, lock, redis_client):
        """get_lock_holder should return None when no lock exists."""
        job_id = uuid4()
        holder = await lock.get_lock_holder(job_id)
        assert holder is None


class TestModuleLevelFunctions:
    """Test module-level convenience functions (design document API)."""

    async def test_acquire_lock_function(self, redis_client):
        """Module-level acquire_lock should work like the class method."""
        job_id = uuid4()
        worker_id = uuid4()
        ttl = 330

        result = await acquire_lock(job_id, worker_id, ttl, redis_client=redis_client)
        assert result is True

        stored = await redis_client.get(_lock_key(job_id))
        assert stored == str(worker_id)

    async def test_release_lock_function(self, redis_client):
        """Module-level release_lock should work like the class method."""
        job_id = uuid4()
        worker_id = uuid4()
        ttl = 330

        await acquire_lock(job_id, worker_id, ttl, redis_client=redis_client)
        result = await release_lock(job_id, worker_id, redis_client=redis_client)
        assert result is True

    async def test_release_lock_function_wrong_worker(self, redis_client):
        """Module-level release_lock should fail for wrong worker."""
        job_id = uuid4()
        worker_a = uuid4()
        worker_b = uuid4()
        ttl = 330

        await acquire_lock(job_id, worker_a, ttl, redis_client=redis_client)
        result = await release_lock(job_id, worker_b, redis_client=redis_client)
        assert result is False


class TestLockTTLBuffer:
    """Test LOCK_TTL_BUFFER constant."""

    def test_buffer_is_30_seconds(self):
        """Lock TTL buffer should be 30 seconds per Requirement 11.2."""
        assert LOCK_TTL_BUFFER == 30
