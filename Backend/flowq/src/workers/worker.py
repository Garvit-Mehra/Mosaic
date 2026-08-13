"""Worker process implementation for the distributed job queue.

Implements Algorithm 2 from the design document: a poll loop that dequeues
jobs from the priority queue, acquires distributed locks, executes handlers
with timeout enforcement, and handles success/failure outcomes.

Requirements covered:
- Req 6.1: Worker dequeues and acquires lock before execution
- Req 6.2: Lock acquisition failure → skip job
- Req 6.3: Update to RUNNING with worker_id and started_at
- Req 6.4: On success → COMPLETED with result and completed_at
- Req 6.5: Timeout → FAILED, trigger retry/DLQ
- Req 6.7: Unregistered handler → FAILED
- Req 6.8: Continue heartbeats during execution
- Req 15.3: Workers stop dequeuing new jobs when PostgreSQL is down
- Req 15.6: Held results are persisted when PostgreSQL connectivity restores
"""

import asyncio
import logging
import os
import signal
import socket
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional
from uuid import UUID

import redis.asyncio as aioredis
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.core.distributed_lock import DistributedLock, LOCK_TTL_BUFFER
from src.core.failure_handler import handle_job_failure
from src.core.handler_registry import HandlerRegistry
from src.core.priority_queue import PriorityQueueInterface
from src.core.state_machine import apply_transition
from src.models.enums import JobStatus, WorkerStatus
from src.models.job import Job
from src.models.job_execution import JobExecution
from src.models.worker import Worker as WorkerModel

logger = logging.getLogger(__name__)

# Heartbeat interval in seconds
HEARTBEAT_INTERVAL = 5
# Redis heartbeat TTL in seconds
HEARTBEAT_TTL = 15
# Dequeue timeout in seconds
DEQUEUE_TIMEOUT = 5.0
# Redis reconnection backoff settings (Req 15.2)
REDIS_RECONNECT_BACKOFF_START = 1.0  # Initial backoff in seconds
REDIS_RECONNECT_BACKOFF_CAP = 30.0  # Maximum backoff in seconds
# PostgreSQL reconnection backoff settings (Req 15.3)
PG_RECONNECT_BACKOFF_START = 1.0  # Initial backoff in seconds
PG_RECONNECT_BACKOFF_CAP = 30.0  # Maximum backoff in seconds
# Interval between pending results flush attempts
PG_FLUSH_INTERVAL = 2.0


@dataclass
class PendingJobResult:
    """Holds a job result in memory when PostgreSQL is unavailable (Req 15.3, 15.6).

    When a job completes execution but the result cannot be persisted
    to PostgreSQL (because PG is down), the result is stored here and
    flushed when connectivity restores.
    """
    job_id: UUID
    status: JobStatus
    result: Optional[dict] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    worker_id: Optional[UUID] = None
    attempt_number: int = 1


class Worker:
    """Worker process that polls for jobs and executes them.

    Follows Algorithm 2 from the design document:
    1. Send heartbeat
    2. Blocking dequeue from priority queue
    3. Acquire distributed lock
    4. Update job to RUNNING
    5. Execute handler with timeout
    6. Handle success/failure
    7. Release lock (always, in finally)
    """

    def __init__(
        self,
        worker_id: UUID,
        session_factory: async_sessionmaker[AsyncSession],
        redis_client,
        priority_queue: PriorityQueueInterface,
        distributed_lock: DistributedLock,
        handler_registry: HandlerRegistry,
    ) -> None:
        """Initialize the worker.

        Args:
            worker_id: Unique identifier for this worker.
            session_factory: Async SQLAlchemy session factory.
            redis_client: Async Redis client for heartbeats.
            priority_queue: Priority queue for job dequeuing.
            distributed_lock: Lock manager for mutual exclusion.
            handler_registry: Registry mapping job_type to handlers.
        """
        self.worker_id = worker_id
        self._session_factory = session_factory
        self._redis = redis_client
        self._priority_queue = priority_queue
        self._distributed_lock = distributed_lock
        self._handler_registry = handler_registry

        self._shutdown_requested = False
        self._current_job_id: Optional[UUID] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._pg_flush_task: Optional[asyncio.Task] = None
        self.jobs_completed: int = 0
        self.jobs_failed: int = 0

        # PostgreSQL failure handling (Req 15.3, 15.6)
        self._pg_available: bool = True
        self._pending_results: List[PendingJobResult] = []

    async def register(self) -> None:
        """Register this worker in PostgreSQL (Req 7.3).

        Creates a Worker record with status ACTIVE, current hostname,
        PID, and sets last_heartbeat to current time.
        """
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        worker_record = WorkerModel(
            id=self.worker_id,
            hostname=socket.gethostname(),
            pid=os.getpid(),
            status=WorkerStatus.ACTIVE,
            last_heartbeat=now,
            started_at=now,
        )
        async with self._session_factory() as session:
            async with session.begin():
                session.add(worker_record)
        logger.info(
            f"Worker {self.worker_id} registered (hostname={worker_record.hostname}, pid={worker_record.pid})"
        )

    async def start(self) -> None:
        """Start the worker poll loop.

        Registers in PostgreSQL, sends initial heartbeat, registers signal
        handlers for graceful shutdown, and then enters the main poll loop.
        The heartbeat runs concurrently as a background task.

        Signal handlers (Req 6.6):
        - SIGTERM: triggers graceful_shutdown()
        - SIGINT: triggers graceful_shutdown()
        """
        self._shutdown_requested = False
        # Register worker in PostgreSQL (Req 7.3)
        await self.register()
        # Register signal handlers for graceful shutdown (Req 6.6)
        self._register_signal_handlers()
        # Start concurrent heartbeat task (Req 6.8)
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        # Start PG pending results flush task (Req 15.6)
        self._pg_flush_task = asyncio.create_task(self._pg_flush_loop())
        try:
            await self._poll_loop()
        finally:
            # Cancel heartbeat task on exit
            if self._heartbeat_task and not self._heartbeat_task.done():
                self._heartbeat_task.cancel()
                try:
                    await self._heartbeat_task
                except asyncio.CancelledError:
                    pass
            # Cancel PG flush task on exit
            if self._pg_flush_task and not self._pg_flush_task.done():
                self._pg_flush_task.cancel()
                try:
                    await self._pg_flush_task
                except asyncio.CancelledError:
                    pass

    async def stop(self, graceful: bool = True) -> None:
        """Stop the worker.

        Args:
            graceful: If True, finish current job before stopping.
                     If False, stop immediately.
        """
        self._shutdown_requested = True
        if not graceful and self._heartbeat_task:
            self._heartbeat_task.cancel()

    def _register_signal_handlers(self) -> None:
        """Register SIGTERM and SIGINT signal handlers for graceful shutdown.

        Uses asyncio loop.add_signal_handler() to schedule graceful_shutdown()
        as a coroutine when a signal is received (Req 6.6).

        On Windows, signal handlers are not supported via add_signal_handler,
        so we fall back to signal.signal() which is less ideal but functional.
        """
        loop = asyncio.get_event_loop()
        if sys.platform != "win32":
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(
                    sig,
                    lambda: asyncio.ensure_future(self.graceful_shutdown()),
                )
        else:
            # Fallback for Windows
            for sig in (signal.SIGTERM, signal.SIGINT):
                signal.signal(
                    sig,
                    lambda s, f: asyncio.ensure_future(self.graceful_shutdown()),
                )
        logger.info(f"Worker {self.worker_id}: Signal handlers registered (SIGTERM, SIGINT)")

    async def graceful_shutdown(self) -> None:
        """Perform graceful shutdown of the worker (Req 6.6, 7.4).

        Steps:
        1. Set _shutdown_requested = True (stops accepting new jobs)
        2. Update worker status to SHUTTING_DOWN in PostgreSQL
        3. Wait for current job to finish (poll loop exits after current iteration)
        4. After current job completes (or no job executing):
           - Remove heartbeat key from Redis
           - Update worker status to IDLE in PostgreSQL
        """
        if self._shutdown_requested:
            # Already shutting down, avoid double-invocation
            return

        logger.info(f"Worker {self.worker_id}: Graceful shutdown initiated")

        # Step 1: Stop accepting new jobs
        self._shutdown_requested = True

        # Step 2: Update worker status to SHUTTING_DOWN in PostgreSQL
        try:
            async with self._session_factory() as session:
                async with session.begin():
                    stmt = select(WorkerModel).where(WorkerModel.id == self.worker_id)
                    result = await session.execute(stmt)
                    worker_record = result.scalar_one_or_none()
                    if worker_record is not None:
                        worker_record.status = WorkerStatus.SHUTTING_DOWN
        except Exception as e:
            logger.warning(f"Worker {self.worker_id}: Failed to update status to SHUTTING_DOWN: {e}")

        # Step 3: Wait for current job to finish
        # The poll loop will exit after the current job completes because
        # _shutdown_requested is True. We wait until _current_job_id is cleared.
        while self._current_job_id is not None:
            await asyncio.sleep(0.1)

        # Step 4: Remove heartbeat key from Redis
        heartbeat_key = f"heartbeat:{self.worker_id}"
        try:
            await self._redis.delete(heartbeat_key)
            logger.info(f"Worker {self.worker_id}: Heartbeat key removed from Redis")
        except Exception as e:
            logger.warning(f"Worker {self.worker_id}: Failed to remove heartbeat key: {e}")

        # Step 5: Update worker status to IDLE in PostgreSQL (shutdown complete)
        try:
            async with self._session_factory() as session:
                async with session.begin():
                    stmt = select(WorkerModel).where(WorkerModel.id == self.worker_id)
                    result = await session.execute(stmt)
                    worker_record = result.scalar_one_or_none()
                    if worker_record is not None:
                        worker_record.status = WorkerStatus.IDLE
                        worker_record.current_job_id = None
        except Exception as e:
            logger.warning(f"Worker {self.worker_id}: Failed to update final status: {e}")

        logger.info(f"Worker {self.worker_id}: Graceful shutdown complete")

    async def _poll_loop(self) -> None:
        """Main poll loop (Algorithm 2) with Redis reconnection (Req 15.2) and PG health (Req 15.3).

        While not shutdown:
        1. Check PG health — if down, pause dequeuing
        2. Dequeue job from priority queue (BZPOPMIN with timeout)
        3. If None → continue
        4. Acquire distributed lock
        5. If lock not acquired → skip, continue
        6. Execute job

        If Redis becomes unreachable during dequeue, the worker pauses polling
        and retries with exponential backoff starting at 1s and capped at 30s.
        Backoff resets on successful dequeue.

        If PostgreSQL becomes unreachable (Req 15.3), the worker stops dequeuing
        new jobs and waits with exponential backoff until PG connectivity restores.
        """
        redis_backoff = REDIS_RECONNECT_BACKOFF_START
        pg_backoff = PG_RECONNECT_BACKOFF_START
        while not self._shutdown_requested:
            # Req 15.3: Check PG health before dequeuing
            if not self._pg_available:
                # PG is down — wait and retry connectivity check
                pg_healthy = await self._check_pg_health()
                if not pg_healthy:
                    logger.warning(
                        f"Worker {self.worker_id}: PostgreSQL unavailable, "
                        f"pausing dequeue. Retrying in {pg_backoff:.1f}s"
                    )
                    await asyncio.sleep(pg_backoff)
                    pg_backoff = min(pg_backoff * 2, PG_RECONNECT_BACKOFF_CAP)
                    continue
                else:
                    # PG recovered
                    logger.info(
                        f"Worker {self.worker_id}: PostgreSQL connectivity restored, "
                        f"resuming dequeue"
                    )
                    self._pg_available = True
                    pg_backoff = PG_RECONNECT_BACKOFF_START

            # Step 1: Dequeue job (blocking pop with 5s timeout)
            try:
                job_id = await self._priority_queue.dequeue(timeout=DEQUEUE_TIMEOUT)
            except (aioredis.ConnectionError, aioredis.TimeoutError, OSError) as e:
                # Req 15.2: Redis unreachable, pause with exponential backoff
                logger.warning(
                    f"Worker {self.worker_id}: Redis connection failed during dequeue: {e}. "
                    f"Retrying in {redis_backoff:.1f}s"
                )
                await asyncio.sleep(redis_backoff)
                redis_backoff = min(redis_backoff * 2, REDIS_RECONNECT_BACKOFF_CAP)
                continue

            # Reset backoff on successful dequeue attempt (even if no job returned)
            redis_backoff = REDIS_RECONNECT_BACKOFF_START

            if job_id is None:
                # No job available, loop back
                continue

            # Step 2: Execute the job (handles lock, execution, cleanup)
            await self.execute_job(job_id)

    async def execute_job(self, job_id: UUID) -> None:
        """Execute a single job with timeout enforcement.

        Acquires lock, fetches job, executes handler, handles result.
        Lock is always released in the finally block.

        Args:
            job_id: The UUID of the job to execute.
        """
        lock_acquired = False
        try:
            # Step 3: Fetch job to determine timeout for lock TTL
            async with self._session_factory() as session:
                stmt = select(Job).where(Job.id == job_id)
                result = await session.execute(stmt)
                job = result.scalar_one_or_none()

            if job is None:
                logger.warning(f"Job {job_id} not found in database, skipping")
                return

            # Step 3: Acquire distributed lock (TTL = timeout + 30s buffer)
            lock_ttl = job.timeout_seconds + LOCK_TTL_BUFFER
            lock_acquired = await self._distributed_lock.acquire_lock(
                job_id, self.worker_id, lock_ttl
            )

            if not lock_acquired:
                # Req 6.2: Lock acquisition failure → skip job
                logger.debug(
                    f"Failed to acquire lock for job {job_id}, skipping"
                )
                return

            # Set current job tracking
            self._current_job_id = job_id

            # Check handler registration (Req 6.7)
            handler = self._handler_registry.get(job.job_type)
            if handler is None:
                error_msg = f"Unregistered job type: {job.job_type}"
                logger.error(f"Job {job_id}: {error_msg}")
                async with self._session_factory() as session:
                    async with session.begin():
                        stmt = select(Job).where(Job.id == job_id)
                        result = await session.execute(stmt)
                        job = result.scalar_one_or_none()
                        if job is None:
                            return
                        # Transition to RUNNING first (needed for failure handler)
                        apply_transition(job, JobStatus.RUNNING)
                        job.worker_id = self.worker_id
                        job.started_at = datetime.now(timezone.utc).replace(tzinfo=None)
                        # Now handle as failure
                        await handle_job_failure(job, error_msg, session, self._redis)
                self.jobs_failed += 1
                return

            # Step 4: Update job to RUNNING (Req 6.3)
            started_at = datetime.now(timezone.utc).replace(tzinfo=None)
            async with self._session_factory() as session:
                async with session.begin():
                    stmt = select(Job).where(Job.id == job_id)
                    result = await session.execute(stmt)
                    job = result.scalar_one_or_none()
                    if job is None:
                        return
                    apply_transition(job, JobStatus.RUNNING)
                    job.worker_id = self.worker_id
                    job.started_at = started_at

            # Step 5: Execute handler with timeout (Req 6.5)
            try:
                handler_result = await asyncio.wait_for(
                    handler(job.payload),
                    timeout=job.timeout_seconds,
                )

                # Step 6a: Success (Req 6.4)
                # If PG fails here, hold result in memory (Req 15.3)
                completed_at = datetime.now(timezone.utc).replace(tzinfo=None)
                try:
                    async with self._session_factory() as session:
                        async with session.begin():
                            stmt = select(Job).where(Job.id == job_id)
                            result = await session.execute(stmt)
                            job = result.scalar_one_or_none()
                            if job is None:
                                return
                            apply_transition(job, JobStatus.COMPLETED)
                            job.result = handler_result if isinstance(handler_result, dict) else {"result": handler_result}
                            job.completed_at = completed_at

                            # Record execution history
                            execution = JobExecution(
                                job_id=job_id,
                                worker_id=self.worker_id,
                                attempt_number=job.retry_count + 1,
                                status="COMPLETED",
                                started_at=started_at,
                                completed_at=completed_at,
                                duration_ms=int(
                                    (completed_at - started_at).total_seconds() * 1000
                                ),
                            )
                            session.add(execution)

                    self.jobs_completed += 1
                    logger.info(f"Job {job_id} completed successfully")
                except Exception as pg_err:
                    # Req 15.3, 15.6: PG failed during result persistence
                    # Hold result in memory for later flush
                    logger.warning(
                        f"Job {job_id}: PostgreSQL failed during result persistence: {pg_err}. "
                        f"Holding result in memory."
                    )
                    self._pg_available = False
                    self._pending_results.append(
                        PendingJobResult(
                            job_id=job_id,
                            status=JobStatus.COMPLETED,
                            result=handler_result if isinstance(handler_result, dict) else {"result": handler_result},
                            started_at=started_at,
                            completed_at=completed_at,
                            worker_id=self.worker_id,
                            attempt_number=job.retry_count + 1 if job else 1,
                        )
                    )
                    self.jobs_completed += 1

            except asyncio.TimeoutError:
                # Step 6b: Timeout (Req 6.5)
                logger.warning(
                    f"Job {job_id} timed out after {job.timeout_seconds}s"
                )
                try:
                    async with self._session_factory() as session:
                        async with session.begin():
                            stmt = select(Job).where(Job.id == job_id)
                            result = await session.execute(stmt)
                            job = result.scalar_one_or_none()
                            if job is None:
                                return
                            await handle_job_failure(
                                job,
                                f"Job exceeded timeout of {job.timeout_seconds} seconds",
                                session,
                                self._redis,
                            )
                    self.jobs_failed += 1
                except Exception as pg_err:
                    # PG failed during failure handling — hold as failed result
                    logger.warning(
                        f"Job {job_id}: PostgreSQL failed during failure persistence: {pg_err}. "
                        f"Holding result in memory."
                    )
                    self._pg_available = False
                    self._pending_results.append(
                        PendingJobResult(
                            job_id=job_id,
                            status=JobStatus.FAILED,
                            error=f"Job exceeded timeout of {job.timeout_seconds} seconds",
                            started_at=started_at,
                            completed_at=datetime.now(timezone.utc).replace(tzinfo=None),
                            worker_id=self.worker_id,
                            attempt_number=job.retry_count + 1 if job else 1,
                        )
                    )
                    self.jobs_failed += 1

            except Exception as e:
                # Step 6c: Execution error
                logger.error(f"Job {job_id} failed with error: {e}")
                try:
                    async with self._session_factory() as session:
                        async with session.begin():
                            stmt = select(Job).where(Job.id == job_id)
                            result = await session.execute(stmt)
                            job = result.scalar_one_or_none()
                            if job is None:
                                return
                            await handle_job_failure(
                                job, str(e), session, self._redis
                            )
                    self.jobs_failed += 1
                except Exception as pg_err:
                    # PG failed during failure handling — hold as failed result
                    logger.warning(
                        f"Job {job_id}: PostgreSQL failed during failure persistence: {pg_err}. "
                        f"Holding result in memory."
                    )
                    self._pg_available = False
                    self._pending_results.append(
                        PendingJobResult(
                            job_id=job_id,
                            status=JobStatus.FAILED,
                            error=str(e),
                            started_at=started_at,
                            completed_at=datetime.now(timezone.utc).replace(tzinfo=None),
                            worker_id=self.worker_id,
                            attempt_number=job.retry_count + 1 if job else 1,
                        )
                    )
                    self.jobs_failed += 1

        finally:
            # Step 7: Always release lock (Req 6.1)
            if lock_acquired:
                await self._distributed_lock.release_lock(job_id, self.worker_id)
            # Clear current job tracking
            self._current_job_id = None

    async def send_heartbeat(self) -> None:
        """Update heartbeat in Redis.

        Sets heartbeat:{worker_id} with current timestamp and TTL of 15s.
        """
        heartbeat_key = f"heartbeat:{self.worker_id}"
        timestamp = str(int(datetime.now(timezone.utc).replace(tzinfo=None).timestamp()))
        try:
            await self._redis.set(heartbeat_key, timestamp, ex=HEARTBEAT_TTL)
        except Exception as e:
            # Req 7.5: Retry on next interval, don't stop execution
            logger.warning(f"Failed to send heartbeat: {e}")

    async def _heartbeat_loop(self) -> None:
        """Background task that sends heartbeats every 5 seconds (Req 6.8).

        Runs concurrently with job execution to ensure heartbeats
        continue during long-running jobs.
        """
        # Send initial heartbeat immediately
        await self.send_heartbeat()
        while not self._shutdown_requested:
            await asyncio.sleep(HEARTBEAT_INTERVAL)
            await self.send_heartbeat()

    async def _check_pg_health(self) -> bool:
        """Check if PostgreSQL is reachable by executing SELECT 1.

        Returns:
            True if PG is healthy, False otherwise.
        """
        try:
            async with self._session_factory() as session:
                await session.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    async def _pg_flush_loop(self) -> None:
        """Background task that flushes pending results to PG when connectivity restores (Req 15.6).

        Periodically checks if there are pending results held in memory
        and attempts to persist them to PostgreSQL. When PG is available
        and all results are flushed, the worker resumes normal dequeuing.
        """
        while not self._shutdown_requested:
            await asyncio.sleep(PG_FLUSH_INTERVAL)

            if not self._pending_results:
                continue

            # Attempt to flush pending results
            if not await self._check_pg_health():
                continue

            # PG is back — flush all pending results
            await self._flush_pending_results()

    async def _flush_pending_results(self) -> None:
        """Persist all held in-memory results to PostgreSQL (Req 15.6).

        Iterates through pending_results and persists each one.
        Successfully persisted results are removed from the list.
        If PG fails again during flush, remaining results stay in memory.
        """
        flushed: List[int] = []
        for idx, pending in enumerate(self._pending_results):
            try:
                async with self._session_factory() as session:
                    async with session.begin():
                        stmt = select(Job).where(Job.id == pending.job_id)
                        result = await session.execute(stmt)
                        job = result.scalar_one_or_none()

                        if job is None:
                            # Job doesn't exist (shouldn't happen), mark as flushed
                            flushed.append(idx)
                            continue

                        if pending.status == JobStatus.COMPLETED:
                            # Only apply transition if job is still in RUNNING state
                            if job.status == JobStatus.RUNNING:
                                apply_transition(job, JobStatus.COMPLETED)
                                job.result = pending.result
                                job.completed_at = pending.completed_at

                                # Record execution history
                                execution = JobExecution(
                                    job_id=pending.job_id,
                                    worker_id=pending.worker_id,
                                    attempt_number=pending.attempt_number,
                                    status="COMPLETED",
                                    started_at=pending.started_at,
                                    completed_at=pending.completed_at,
                                    duration_ms=int(
                                        (pending.completed_at - pending.started_at).total_seconds() * 1000
                                    ) if pending.started_at and pending.completed_at else None,
                                )
                                session.add(execution)
                        elif pending.status == JobStatus.FAILED:
                            if job.status == JobStatus.RUNNING:
                                await handle_job_failure(
                                    job, pending.error or "Unknown error", session, self._redis
                                )

                flushed.append(idx)
                logger.info(
                    f"Worker {self.worker_id}: Flushed pending result for job {pending.job_id}"
                )
            except Exception as e:
                # PG failed again during flush — stop flushing, keep remaining
                logger.warning(
                    f"Worker {self.worker_id}: PG failed during flush of job {pending.job_id}: {e}"
                )
                break

        # Remove successfully flushed entries (in reverse to preserve indices)
        for idx in reversed(flushed):
            self._pending_results.pop(idx)

        if not self._pending_results:
            # All flushed — mark PG as available and resume dequeuing
            self._pg_available = True
            logger.info(
                f"Worker {self.worker_id}: All pending results flushed. "
                f"PostgreSQL connectivity restored, resuming normal operation."
            )
