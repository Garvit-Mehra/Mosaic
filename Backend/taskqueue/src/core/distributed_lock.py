"""Distributed lock implementation using Redis SET NX with TTL.

Provides mutual exclusion for job execution — at most one worker holds
the lock for a given job at any time. Uses an atomic Lua script for
safe compare-and-delete release.

Key format: lock:job:{job_id}
Value: worker_id (UUID string)
TTL: job timeout_seconds + 30 seconds buffer
"""

from typing import Optional
from uuid import UUID

import redis.asyncio as redis

from src.redis_client import get_redis_client

# Lua script for atomic compare-and-delete.
# Only deletes the key if the current value matches the provided worker_id.
# Returns 1 if deleted (lock released), 0 if value didn't match or key missing.
RELEASE_LOCK_SCRIPT = """
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
else
    return 0
end
"""

# Buffer added to job timeout for lock TTL (seconds)
LOCK_TTL_BUFFER = 30


def _lock_key(job_id: UUID) -> str:
    """Build the Redis key for a job's distributed lock."""
    return f"lock:job:{job_id}"


class DistributedLock:
    """Redis-backed distributed lock for job execution mutual exclusion.

    Ensures at most one worker executes a given job at any time. The lock
    auto-expires after TTL to prevent deadlocks when a worker crashes.
    """

    def __init__(self, redis_client: Optional[redis.Redis] = None):
        """Initialize the distributed lock manager.

        Args:
            redis_client: Optional Redis client instance. Falls back to the
                shared client from src/redis_client.py if not provided.
        """
        self._redis = redis_client

    @property
    def redis(self) -> redis.Redis:
        """Get the Redis client, falling back to the shared instance."""
        if self._redis is None:
            self._redis = get_redis_client()
        return self._redis

    async def acquire_lock(self, job_id: UUID, worker_id: UUID, ttl: int) -> bool:
        """Acquire a distributed lock for a job using Redis SET NX.

        The lock key is set only if it does not already exist (NX flag),
        ensuring mutual exclusion. The TTL should be job timeout_seconds + 30
        to accommodate execution duration.

        Args:
            job_id: The job to lock.
            worker_id: The worker acquiring the lock (stored as lock value).
            ttl: Lock time-to-live in seconds (timeout_seconds + LOCK_TTL_BUFFER).

        Returns:
            True if the lock was acquired, False if already held by another worker.
        """
        lock_key = _lock_key(job_id)
        lock_value = str(worker_id)

        # SET NX (only if not exists) with expiry
        acquired = await self.redis.set(lock_key, lock_value, nx=True, ex=ttl)
        return acquired is not None

    async def release_lock(self, job_id: UUID, worker_id: UUID) -> bool:
        """Release a distributed lock using atomic compare-and-delete.

        Only the holder (matching worker_id) can release the lock. If the lock
        has already expired or is held by a different worker, this returns False.

        When this returns False, the caller should discard its execution result
        because the lock expired while the job was still running (Requirement 11.5).

        Args:
            job_id: The job whose lock to release.
            worker_id: The worker attempting to release (must match lock value).

        Returns:
            True if the lock was successfully released by this worker.
            False if the lock was not held by this worker (expired or reassigned).
        """
        lock_key = _lock_key(job_id)
        lock_value = str(worker_id)

        result = await self.redis.eval(RELEASE_LOCK_SCRIPT, 1, lock_key, lock_value)
        return result == 1

    async def is_locked(self, job_id: UUID) -> bool:
        """Check if a job currently has a lock held.

        Args:
            job_id: The job to check.

        Returns:
            True if the lock key exists in Redis.
        """
        lock_key = _lock_key(job_id)
        return await self.redis.exists(lock_key) > 0

    async def get_lock_holder(self, job_id: UUID) -> Optional[str]:
        """Get the worker_id currently holding the lock for a job.

        Args:
            job_id: The job to check.

        Returns:
            The worker_id string if locked, None if not locked.
        """
        lock_key = _lock_key(job_id)
        return await self.redis.get(lock_key)


# Module-level convenience functions that match the design document's API

async def acquire_lock(
    job_id: UUID, worker_id: UUID, ttl: int, redis_client: Optional[redis.Redis] = None
) -> bool:
    """Acquire a distributed lock for a job using Redis SET NX.

    Convenience function matching the design document's Algorithm 6 interface.

    Args:
        job_id: The job to lock.
        worker_id: The worker acquiring the lock.
        ttl: Lock TTL in seconds (should be timeout_seconds + 30).
        redis_client: Optional Redis client. Uses shared client if not provided.

    Returns:
        True if the lock was acquired, False if already held.
    """
    lock = DistributedLock(redis_client)
    return await lock.acquire_lock(job_id, worker_id, ttl)


async def release_lock(
    job_id: UUID, worker_id: UUID, redis_client: Optional[redis.Redis] = None
) -> bool:
    """Release a distributed lock using atomic compare-and-delete.

    Convenience function matching the design document's Algorithm 6 interface.
    Returns False when the lock was not held by this worker — in which case the
    caller should discard its execution result (Requirement 11.5).

    Args:
        job_id: The job whose lock to release.
        worker_id: The worker attempting to release.
        redis_client: Optional Redis client. Uses shared client if not provided.

    Returns:
        True if released by this worker, False if not held (expired/reassigned).
    """
    lock = DistributedLock(redis_client)
    return await lock.release_lock(job_id, worker_id)
