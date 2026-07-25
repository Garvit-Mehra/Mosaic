"""State reconstruction from PostgreSQL on startup or Redis recovery.

Rebuilds all Redis queue data structures (priority queue, scheduled set, DLQ)
from the authoritative PostgreSQL state. This ensures the system can recover
fully after a restart or Redis failure.

Requirements covered:
- 13.2: Rebuild Redis state from PG on restart
- 13.4: Redis state is fully reconstructable from PG
- 13.5: Block new submissions until reconstruction completes
- 15.4: On Redis recovery, reconcile within 60 seconds
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

import redis.asyncio as redis
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.core.priority_queue import (
    QUEUE_KEY,
    PriorityQueueInterface,
    calculate_queue_score,
)
from src.models.enums import JobStatus
from src.models.job import Job

logger = logging.getLogger(__name__)

# Redis keys managed during reconstruction
SCHEDULED_QUEUE_KEY = "queue:scheduled"
RETRY_QUEUE_KEY = "queue:retry"
DLQ_KEY = "queue:dlq"

# Maximum time allowed for reconciliation after Redis recovery (Req 15.4)
RECONCILIATION_TIMEOUT_SECONDS = 60


class StateReconstructor:
    """Rebuilds Redis queue state from PostgreSQL.

    On startup or Redis recovery, this class:
    1. Clears existing Redis queue keys to start fresh
    2. Rebuilds queue:priority from all QUEUED jobs
    3. Rebuilds queue:scheduled from all SCHEDULED jobs with future execute_at
    4. Treats RUNNING jobs as abandoned (re-queue or DLQ based on retry count)

    The reconstruction process blocks new submissions and queue operations
    until fully complete (Req 13.5).

    Args:
        session_factory: Async SQLAlchemy session factory for database access.
        redis_client: Async Redis client for queue operations.
        priority_queue: Priority queue implementation for score calculation.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        redis_client: redis.Redis,
        priority_queue: PriorityQueueInterface,
    ) -> None:
        self._session_factory = session_factory
        self._redis = redis_client
        self._priority_queue = priority_queue
        self._reconstruction_complete = asyncio.Event()
        self._is_reconstructing = False

    @property
    def is_ready(self) -> bool:
        """Whether reconstruction has completed and the system is ready."""
        return self._reconstruction_complete.is_set()

    async def wait_until_ready(self, timeout: Optional[float] = None) -> bool:
        """Block until reconstruction completes.

        Args:
            timeout: Maximum seconds to wait. None means wait indefinitely.

        Returns:
            True if reconstruction completed, False if timeout elapsed.
        """
        try:
            await asyncio.wait_for(
                self._reconstruction_complete.wait(), timeout=timeout
            )
            return True
        except asyncio.TimeoutError:
            return False

    async def reconstruct(self) -> dict:
        """Rebuild all Redis state from PostgreSQL.

        Deletes existing Redis queue keys and rebuilds them from the
        authoritative PostgreSQL state. Blocks submissions until complete.

        Returns:
            Dict with counts: priority_queue_rebuilt, scheduled_rebuilt,
            running_recovered, total_time_ms.
        """
        self._is_reconstructing = True
        self._reconstruction_complete.clear()
        start_time = datetime.now(timezone.utc)

        logger.info("Starting state reconstruction from PostgreSQL...")

        try:
            # Step 1: Clear existing Redis queue keys to start fresh
            await self._clear_redis_queues()

            # Step 2: Rebuild priority queue from QUEUED jobs
            priority_count = await self._rebuild_priority_queue()

            # Step 3: Rebuild scheduled queue from SCHEDULED jobs
            scheduled_count = await self._rebuild_scheduled_queue()

            # Step 4: Recover abandoned RUNNING jobs
            recovered_count = await self._recover_running_jobs()

            elapsed_ms = int(
                (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            )

            result = {
                "priority_queue_rebuilt": priority_count,
                "scheduled_rebuilt": scheduled_count,
                "running_recovered": recovered_count,
                "total_time_ms": elapsed_ms,
            }

            logger.info(
                "State reconstruction complete in %dms: "
                "%d priority, %d scheduled, %d recovered",
                elapsed_ms,
                priority_count,
                scheduled_count,
                recovered_count,
            )

            return result

        finally:
            # Signal that reconstruction is complete (Req 13.5)
            self._is_reconstructing = False
            self._reconstruction_complete.set()

    async def reconcile(self) -> dict:
        """Reconcile Redis state from PostgreSQL after Redis recovery.

        Same as reconstruct() but enforces the 60-second timeout (Req 15.4).

        Returns:
            Dict with reconstruction counts.

        Raises:
            asyncio.TimeoutError: If reconciliation exceeds 60 seconds.
        """
        logger.info(
            "Starting Redis reconciliation (timeout=%ds)...",
            RECONCILIATION_TIMEOUT_SECONDS,
        )
        return await asyncio.wait_for(
            self.reconstruct(), timeout=RECONCILIATION_TIMEOUT_SECONDS
        )

    async def _clear_redis_queues(self) -> None:
        """Delete existing Redis queue keys to start fresh."""
        keys_to_delete = [QUEUE_KEY, SCHEDULED_QUEUE_KEY, RETRY_QUEUE_KEY, DLQ_KEY]
        for key in keys_to_delete:
            await self._redis.delete(key)
        logger.debug("Cleared Redis queue keys: %s", keys_to_delete)

    async def _rebuild_priority_queue(self) -> int:
        """Rebuild queue:priority from all QUEUED jobs in PostgreSQL.

        Queries all jobs with status=QUEUED from PostgreSQL and adds them
        to the Redis priority sorted set with the appropriate score.

        Returns:
            Number of jobs added to the priority queue.
        """
        count = 0
        async with self._session_factory() as session:
            stmt = select(Job).where(Job.status == JobStatus.QUEUED)
            result = await session.execute(stmt)
            queued_jobs = result.scalars().all()

            if not queued_jobs:
                logger.debug("No QUEUED jobs to rebuild in priority queue")
                return 0

            # Batch ZADD for efficiency
            members = {}
            for job in queued_jobs:
                score = calculate_queue_score(job.priority, job.created_at)
                members[str(job.id)] = score
                count += 1

            if members:
                await self._redis.zadd(QUEUE_KEY, members)

        logger.debug("Rebuilt priority queue with %d jobs", count)
        return count

    async def _rebuild_scheduled_queue(self) -> int:
        """Rebuild queue:scheduled from all SCHEDULED jobs with future execute_at.

        Queries all jobs with status=SCHEDULED and execute_at > now from
        PostgreSQL and adds them to the Redis schedule sorted set.

        Returns:
            Number of jobs added to the scheduled queue.
        """
        count = 0
        now = datetime.utcnow()

        async with self._session_factory() as session:
            stmt = select(Job).where(
                Job.status == JobStatus.SCHEDULED,
                Job.execute_at > now,
            )
            result = await session.execute(stmt)
            scheduled_jobs = result.scalars().all()

            if not scheduled_jobs:
                logger.debug("No SCHEDULED jobs to rebuild in schedule set")
                return 0

            # Batch ZADD for efficiency
            members = {}
            for job in scheduled_jobs:
                # Score = execute_at timestamp in milliseconds
                execute_at = job.execute_at
                if execute_at.tzinfo is None:
                    execute_at = execute_at.replace(tzinfo=timezone.utc)
                score = int(execute_at.timestamp() * 1000)
                members[str(job.id)] = score
                count += 1

            if members:
                await self._redis.zadd(SCHEDULED_QUEUE_KEY, members)

        logger.debug("Rebuilt scheduled queue with %d jobs", count)
        return count

    async def _recover_running_jobs(self) -> int:
        """Treat RUNNING jobs as abandoned and re-queue or DLQ.

        On startup, any job still in RUNNING state is considered abandoned
        (the system restarted while the job was executing). For each:
        - If retry_count < max_retries: increment retry_count, set QUEUED,
          clear worker_id, add to priority queue.
        - If retry_count >= max_retries: set DEAD_LETTER, clear worker_id,
          add to DLQ.

        Returns:
            Number of jobs recovered.
        """
        count = 0
        now = datetime.utcnow()

        async with self._session_factory() as session:
            async with session.begin():
                stmt = select(Job).where(Job.status == JobStatus.RUNNING)
                result = await session.execute(stmt)
                running_jobs = result.scalars().all()

                if not running_jobs:
                    logger.debug("No RUNNING jobs to recover")
                    return 0

                priority_members = {}
                dlq_members = {}

                for job in running_jobs:
                    count += 1
                    new_retry_count = job.retry_count + 1

                    if new_retry_count <= job.max_retries:
                        # Re-queue: increment retry, set QUEUED
                        job.status = JobStatus.QUEUED
                        job.retry_count = new_retry_count
                        job.worker_id = None
                        job.started_at = None
                        job.error = "Job abandoned during system restart"
                        job.updated_at = now

                        score = calculate_queue_score(job.priority, now)
                        priority_members[str(job.id)] = score
                    else:
                        # DLQ: max retries exceeded
                        job.status = JobStatus.DEAD_LETTER
                        job.worker_id = None
                        job.error = (
                            f"Job abandoned during system restart; "
                            f"max retries ({job.max_retries}) exceeded"
                        )
                        job.updated_at = now

                        dlq_score = int(now.timestamp() * 1000)
                        dlq_members[str(job.id)] = dlq_score

            # Session committed here - PG persisted before Redis

        # Update Redis after PG persistence
        if priority_members:
            await self._redis.zadd(QUEUE_KEY, priority_members)

        if dlq_members:
            await self._redis.zadd(DLQ_KEY, dlq_members)

        logger.debug(
            "Recovered %d running jobs: %d re-queued, %d to DLQ",
            count,
            len(priority_members),
            len(dlq_members),
        )
        return count
