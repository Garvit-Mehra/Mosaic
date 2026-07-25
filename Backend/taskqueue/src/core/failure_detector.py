"""Failure detector for dead worker recovery.

Implements Algorithm 5 from the design document: periodically checks
for workers whose heartbeat has expired in Redis, marks them as DEAD,
and recovers their abandoned RUNNING jobs by either re-queuing (with
incremented retry_count) or moving to the dead-letter queue.

Requirements covered:
- 8.1: Detect worker with expired heartbeat, mark as DEAD
- 8.2: Recover ALL RUNNING jobs from dead worker
- 8.3: Re-queue jobs with retries remaining (increment retry_count)
- 8.4: Move jobs with exhausted retries to DLQ
- 8.5: Check every 5 seconds
- 8.6: Detection within 20 seconds (15s TTL + 5s check)
- 8.7: Only recover RUNNING jobs (prevents double-recovery)
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import List, Optional, Tuple
from uuid import UUID

import redis.asyncio as redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.config import settings
from src.core.priority_queue import PriorityQueueInterface, calculate_queue_score
from src.models.enums import JobStatus, WorkerStatus
from src.models.job import Job
from src.models.worker import Worker

logger = logging.getLogger(__name__)

# Redis key for the dead-letter queue sorted set
DLQ_KEY = "queue:dlq"


class FailureDetector:
    """Monitor worker heartbeats and recover abandoned jobs.

    Runs a background loop every 5 seconds that:
    1. Queries ACTIVE workers from PostgreSQL.
    2. Checks if their heartbeat key exists in Redis.
    3. For dead workers (expired heartbeat): recovers all RUNNING jobs.
    4. Re-queues jobs with retries remaining or moves to DLQ.
    5. Marks the worker as DEAD in PostgreSQL.

    Uses SELECT ... FOR UPDATE to prevent double-recovery when multiple
    detector instances run concurrently (Req 8.7).

    Args:
        session_factory: Async SQLAlchemy session factory for database access.
        redis_client: Async Redis client for heartbeat checks and queue ops.
        priority_queue: Priority queue implementation for re-queuing jobs.
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
        """Start the failure detection loop as a background asyncio task."""
        self._shutdown_requested = False
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Failure detector started")

    async def stop(self) -> None:
        """Stop the failure detection loop gracefully."""
        self._shutdown_requested = True
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=10.0)
            except asyncio.TimeoutError:
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
            self._task = None
        logger.info("Failure detector stopped")

    async def _run_loop(self) -> None:
        """Main failure detection loop: check for dead workers every interval."""
        while not self._shutdown_requested:
            try:
                dead_workers = await self.detect_dead_workers()
                for worker_id in dead_workers:
                    recovered = await self.recover_abandoned_jobs(worker_id)
                    if recovered > 0:
                        logger.info(
                            "Recovered %d jobs from dead worker %s",
                            recovered,
                            worker_id,
                        )
            except Exception:
                logger.exception("Error in failure detection loop")

            # Req 8.5: Check every 5 seconds
            try:
                await asyncio.sleep(settings.failure_check_interval_seconds)
            except asyncio.CancelledError:
                break

    async def detect_dead_workers(self) -> List[UUID]:
        """Find workers with expired heartbeats.

        Queries all ACTIVE workers from PostgreSQL and checks whether
        their heartbeat key exists in Redis. Workers whose heartbeat
        has expired (key is None) are considered dead.

        Returns:
            List of dead worker UUIDs.
        """
        dead_worker_ids: List[UUID] = []

        async with self._session_factory() as session:
            # Query workers with ACTIVE status
            stmt = select(Worker).where(Worker.status == WorkerStatus.ACTIVE)
            result = await session.execute(stmt)
            active_workers = result.scalars().all()

        for worker in active_workers:
            # Check if heartbeat key exists in Redis
            heartbeat = await self._redis.get(f"heartbeat:{worker.id}")
            if heartbeat is None:
                # Heartbeat expired → worker is dead
                dead_worker_ids.append(worker.id)
                logger.warning(
                    "Worker %s heartbeat expired, marking as dead",
                    worker.id,
                )

        return dead_worker_ids

    async def recover_abandoned_jobs(self, worker_id: UUID) -> int:
        """Re-queue or DLQ jobs from a dead worker.

        Finds all RUNNING jobs assigned to the dead worker and either:
        - Re-queues them (if retry_count < max_retries): increment retry_count,
          set status to QUEUED, clear worker_id, add to priority queue.
        - Moves to DLQ (if retry_count >= max_retries): set status to DEAD_LETTER,
          clear worker_id, add to queue:dlq sorted set.

        Uses SELECT FOR UPDATE to ensure only RUNNING jobs are recovered,
        preventing double-recovery by concurrent detector instances (Req 8.7).

        Args:
            worker_id: The UUID of the dead worker whose jobs to recover.

        Returns:
            Count of recovered jobs.
        """
        # Track jobs for Redis operations after PG commit
        requeued_jobs: List[Tuple[UUID, int]] = []  # (job_id, priority)
        dlq_jobs: List[UUID] = []

        async with self._session_factory() as session:
            async with session.begin():
                # Req 8.7: SELECT FOR UPDATE prevents concurrent recovery.
                # Only select jobs that are STILL in RUNNING status.
                stmt = (
                    select(Job)
                    .where(Job.worker_id == worker_id)
                    .where(Job.status == JobStatus.RUNNING)
                    .with_for_update()
                )
                result = await session.execute(stmt)
                abandoned_jobs = result.scalars().all()

                for job in abandoned_jobs:
                    if job.retry_count < job.max_retries:
                        # Req 8.3: Re-queue with incremented retry count
                        job.retry_count += 1
                        job.status = JobStatus.QUEUED
                        job.worker_id = None
                        job.error = f"Worker {worker_id} died during execution"
                        job.updated_at = datetime.now(timezone.utc)
                        requeued_jobs.append((job.id, job.priority))
                    else:
                        # Req 8.4: Move to DLQ
                        job.status = JobStatus.DEAD_LETTER
                        job.worker_id = None
                        job.error = (
                            f"Worker died; max retries ({job.max_retries}) exceeded"
                        )
                        job.updated_at = datetime.now(timezone.utc)
                        dlq_jobs.append(job.id)

                # Req 8.1: Mark worker as DEAD in PostgreSQL
                worker_stmt = (
                    select(Worker)
                    .where(Worker.id == worker_id)
                    .with_for_update()
                )
                worker_result = await session.execute(worker_stmt)
                worker = worker_result.scalar_one_or_none()
                if worker is not None:
                    worker.status = WorkerStatus.DEAD

            # Transaction committed here - PG persisted before Redis

        # Now perform Redis operations after successful PG commit
        for job_id, priority in requeued_jobs:
            score = calculate_queue_score(priority, datetime.now(timezone.utc))
            await self._redis.zadd("queue:priority", {str(job_id): score})

        for job_id in dlq_jobs:
            failed_at_score = int(datetime.now(timezone.utc).timestamp() * 1000)
            await self._redis.zadd(DLQ_KEY, {str(job_id): failed_at_score})

        # Req 8.1: Delete heartbeat key from Redis (cleanup)
        await self._redis.delete(f"heartbeat:{worker_id}")

        recovered_count = len(requeued_jobs) + len(dlq_jobs)
        return recovered_count
