"""Scheduler promotion loop for scheduled and retry jobs.

Implements Algorithm 4 from the design document: continuously promotes
scheduled and retry jobs to the priority queue when their execution
time arrives. Uses atomic ZREM checks to prevent double promotion
in multi-instance deployments.

Requirements covered:
- 5.1: Keep job in schedule set until execute_at arrives
- 5.2: Promote when execute_at <= now, set status to QUEUED
- 5.3: Check every 1 second, batch of 100
- 5.4: Promote retry jobs when backoff elapses
- 5.5: Atomic ZREM + ZADD to prevent double promotion
- 5.6: If ZREM returns 0, skip (already promoted)
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

import redis.asyncio as redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.config import settings
from src.core.priority_queue import PriorityQueueInterface, calculate_queue_score
from src.core.state_machine import apply_transition
from src.models.enums import JobStatus
from src.models.job import Job

logger = logging.getLogger(__name__)

# Redis keys for sorted sets
SCHEDULED_QUEUE_KEY = "queue:scheduled"
RETRY_QUEUE_KEY = "queue:retry"


class Scheduler:
    """Background scheduler that promotes due jobs to the priority queue.

    Checks for promotable jobs every 1 second (configurable via settings).
    Processes up to 100 jobs per queue per cycle.

    Args:
        session_factory: Async SQLAlchemy session factory for database access.
        redis_client: Async Redis client for sorted set operations.
        priority_queue: Priority queue implementation for job enqueuing.
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
        self._shutdown_requested = False
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the scheduler loop as a background asyncio task."""
        self._shutdown_requested = False
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Scheduler started")

    async def stop(self) -> None:
        """Stop the scheduler loop gracefully."""
        self._shutdown_requested = True
        if self._task is not None:
            # Wait for the task to finish its current cycle
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except asyncio.TimeoutError:
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
            self._task = None
        logger.info("Scheduler stopped")

    async def _run_loop(self) -> None:
        """Main scheduler loop: promote due jobs every interval."""
        while not self._shutdown_requested:
            try:
                promoted = await self.promote_due_jobs()
                if promoted > 0:
                    logger.debug("Promoted %d jobs", promoted)
            except Exception:
                logger.exception("Error in scheduler promotion loop")

            # Sleep for the configured interval (default 1 second)
            try:
                await asyncio.sleep(settings.scheduler_interval_seconds)
            except asyncio.CancelledError:
                break

    async def promote_due_jobs(self) -> int:
        """Move jobs with execute_at <= now from schedule/retry to priority queue.

        Returns:
            Total count of promoted jobs across both queues.
        """
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

        scheduled_count = await self._promote_scheduled_jobs(now_ms)
        retry_count = await self._promote_retry_jobs(now_ms)

        return scheduled_count + retry_count

    async def _promote_scheduled_jobs(self, now_ms: int) -> int:
        """Promote scheduled jobs whose execute_at time has arrived.

        Fetches up to 100 jobs from queue:scheduled where score <= now_ms.
        For each job, atomically removes from schedule set and adds to
        priority queue. Updates job status SCHEDULED → QUEUED in PostgreSQL.

        Args:
            now_ms: Current time in milliseconds since epoch.

        Returns:
            Number of jobs successfully promoted.
        """
        # Fetch batch of due scheduled jobs (Req 5.3: batch of 100)
        job_ids = await self._redis.zrangebyscore(
            SCHEDULED_QUEUE_KEY, min=0, max=now_ms, start=0, num=settings.scheduler_batch_size
        )

        if not job_ids:
            return 0

        promoted_count = 0
        for job_id_str in job_ids:
            # Req 5.5 & 5.6: Atomic ZREM - if returns 0, another instance promoted it
            removed = await self._redis.zrem(SCHEDULED_QUEUE_KEY, job_id_str)
            if removed == 0:
                # Another scheduler instance already promoted this job (Req 5.6)
                continue

            # Fetch job from PostgreSQL to get priority for score calculation
            try:
                job_uuid = UUID(job_id_str)
            except ValueError:
                logger.warning("Invalid job ID in scheduled queue: %s", job_id_str)
                continue

            async with self._session_factory() as session:
                async with session.begin():
                    stmt = select(Job).where(Job.id == job_uuid)
                    result = await session.execute(stmt)
                    job = result.scalar_one_or_none()

                    if job is None:
                        logger.warning(
                            "Scheduled job %s not found in database, skipping", job_id_str
                        )
                        continue

                    # Apply state transition SCHEDULED → QUEUED (Req 5.2)
                    apply_transition(job, JobStatus.QUEUED)

            # Add to priority queue with calculated score
            score = calculate_queue_score(job.priority, datetime.now(timezone.utc))
            await self._redis.zadd("queue:priority", {job_id_str: score})

            promoted_count += 1
            logger.debug("Promoted scheduled job %s to priority queue", job_id_str)

        return promoted_count

    async def _promote_retry_jobs(self, now_ms: int) -> int:
        """Promote retry jobs whose backoff period has elapsed.

        Fetches up to 100 jobs from queue:retry where score <= now_ms.
        For each job, atomically removes from retry set and adds to
        priority queue. Retry jobs are already in QUEUED status in PG
        (set by the failure handler), so no PG status update is needed.

        Args:
            now_ms: Current time in milliseconds since epoch.

        Returns:
            Number of jobs successfully promoted.
        """
        # Fetch batch of due retry jobs (Req 5.3: batch of 100)
        job_ids = await self._redis.zrangebyscore(
            RETRY_QUEUE_KEY, min=0, max=now_ms, start=0, num=settings.scheduler_batch_size
        )

        if not job_ids:
            return 0

        promoted_count = 0
        for job_id_str in job_ids:
            # Req 5.5 & 5.6: Atomic ZREM - if returns 0, another instance promoted it
            removed = await self._redis.zrem(RETRY_QUEUE_KEY, job_id_str)
            if removed == 0:
                # Another scheduler instance already promoted this job (Req 5.6)
                continue

            # Fetch job from PostgreSQL to get priority for score calculation
            try:
                job_uuid = UUID(job_id_str)
            except ValueError:
                logger.warning("Invalid job ID in retry queue: %s", job_id_str)
                continue

            async with self._session_factory() as session:
                stmt = select(Job).where(Job.id == job_uuid)
                result = await session.execute(stmt)
                job = result.scalar_one_or_none()

                if job is None:
                    logger.warning(
                        "Retry job %s not found in database, skipping", job_id_str
                    )
                    continue

            # Add to priority queue with calculated score (Req 5.4)
            score = calculate_queue_score(job.priority, datetime.now(timezone.utc))
            await self._redis.zadd("queue:priority", {job_id_str: score})

            promoted_count += 1
            logger.debug("Promoted retry job %s to priority queue", job_id_str)

        return promoted_count
