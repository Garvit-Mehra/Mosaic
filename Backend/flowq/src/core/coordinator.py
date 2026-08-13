"""Job Coordinator: central orchestrator for job lifecycle management.

Implements submit, get, list, and history operations with PostgreSQL as
the source of truth and Redis for queue coordination. All state changes
are validated through the state machine before persistence.
"""

from datetime import datetime, timezone
from typing import List, Optional
from uuid import UUID

import redis.asyncio as redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.core.priority_queue import PriorityQueueInterface, calculate_queue_score
from src.core.state_machine import apply_transition, validate_transition
from src.core.validators import validate_job_submission
from src.models.enums import JobStatus, WorkerStatus
from src.models.job import Job
from src.models.job_execution import JobExecution
from src.models.worker import Worker

# Redis key for the scheduled jobs sorted set
SCHEDULED_QUEUE_KEY = "queue:scheduled"
# Redis key for the dead-letter queue sorted set
DLQ_KEY = "queue:dlq"


class JobCoordinator:
    """Central orchestrator for job lifecycle management.

    Coordinates between PostgreSQL (source of truth) and Redis (queue operations).
    Enforces state machine transitions on all status changes and persists state
    to PostgreSQL before updating Redis.

    Args:
        session_factory: Async SQLAlchemy session factory for database access.
        redis_client: Async Redis client for queue operations.
        priority_queue: Priority queue implementation for immediate job routing.
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

    async def submit_job(
        self,
        job_type: str,
        payload: dict,
        priority: int = 0,
        execute_at: Optional[datetime] = None,
        timeout_seconds: int = 300,
        max_retries: int = 3,
        retry_backoff_base: float = 2.0,
    ) -> dict:
        """Create a new job, persist to PostgreSQL, and route to queue or schedule set.

        Steps:
            1. Validate submission parameters.
            2. Create Job in PostgreSQL with status=PENDING.
            3. If no execute_at: transition PENDING→QUEUED, add to priority queue.
            4. If execute_at set: transition PENDING→SCHEDULED, add to schedule set.
            5. Persist state to PG before Redis operations.

        Args:
            job_type: Registered handler type for the job.
            payload: JSON-serializable job payload.
            priority: Job priority (0-10000, higher = more urgent).
            execute_at: Optional future execution time for scheduled jobs.
            timeout_seconds: Maximum execution time in seconds.
            max_retries: Maximum retry attempts on failure.
            retry_backoff_base: Base for exponential backoff calculation.

        Returns:
            Dict with job id, status, job_type, priority, and created_at.

        Raises:
            ValidationError: If any submission parameter is invalid.
        """
        # Step 1: Validate all submission parameters
        validate_job_submission(
            job_type=job_type,
            payload=payload,
            priority=priority,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            execute_at=execute_at,
        )

        async with self._session_factory() as session:
            async with session.begin():
                # Step 2: Create job in PostgreSQL with PENDING status
                now = datetime.now(timezone.utc).replace(tzinfo=None)
                job = Job(
                    job_type=job_type,
                    payload=payload,
                    priority=priority,
                    status=JobStatus.PENDING,
                    execute_at=execute_at,
                    timeout_seconds=timeout_seconds,
                    max_retries=max_retries,
                    retry_backoff_base=retry_backoff_base,
                    created_at=now,
                    updated_at=now,
                )
                session.add(job)
                # Flush to generate the ID without committing yet
                await session.flush()

                if execute_at is None:
                    # Step 3: Immediate job → PENDING → QUEUED
                    apply_transition(job, JobStatus.QUEUED)
                else:
                    # Step 4: Scheduled job → PENDING → SCHEDULED
                    apply_transition(job, JobStatus.SCHEDULED)

            # Session committed here (PG persisted before Redis)
            # Capture values we need for Redis operations
            job_id = job.id
            job_status = job.status
            job_priority = job.priority
            job_created_at = job.created_at
            job_type_val = job.job_type

        # Step 5: Route to Redis after PG persistence (Req 13.1)
        if execute_at is None:
            # Add to priority queue
            await self._priority_queue.enqueue(
                job_id=job_id,
                priority=job_priority,
                enqueued_at=job_created_at,
            )
        else:
            # Add to scheduled sorted set with score = execute_at timestamp in ms
            if execute_at.tzinfo is None:
                execute_at_aware = execute_at.replace(tzinfo=timezone.utc)
            else:
                execute_at_aware = execute_at
            score = int(execute_at_aware.timestamp() * 1000)
            await self._redis.zadd(
                SCHEDULED_QUEUE_KEY, {str(job_id): score}
            )

        # Return response (Req 1.8)
        return {
            "id": job_id,
            "status": job_status.value,
            "job_type": job_type_val,
            "priority": job_priority,
            "created_at": job_created_at,
        }

    async def get_job(self, job_id: UUID) -> Optional[dict]:
        """Retrieve a job by its ID from PostgreSQL.

        Args:
            job_id: The UUID of the job to retrieve.

        Returns:
            Dict with complete job metadata, or None if not found.
        """
        async with self._session_factory() as session:
            stmt = select(Job).where(Job.id == job_id)
            result = await session.execute(stmt)
            job = result.scalar_one_or_none()

            if job is None:
                return None

            return _job_to_response(job)

    async def list_jobs(
        self,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[dict]:
        """List jobs with optional status filter and pagination.

        Args:
            status: Optional status string to filter by (e.g. "QUEUED").
            limit: Maximum number of results (default 50, capped at 100).
            offset: Number of results to skip (default 0).

        Returns:
            List of job response dicts ordered by created_at descending.
        """
        # Cap limit at 100
        limit = min(limit, 100)
        if limit < 1:
            limit = 1
        if offset < 0:
            offset = 0

        async with self._session_factory() as session:
            stmt = select(Job).order_by(Job.created_at.desc())

            if status is not None:
                # Convert string to enum for filtering
                try:
                    status_enum = JobStatus(status)
                except ValueError:
                    # Invalid status filter - return empty list
                    return []
                stmt = stmt.where(Job.status == status_enum)
            else:
                # Exclude DEAD_LETTER jobs from general listings (Req 14.4)
                stmt = stmt.where(Job.status != JobStatus.DEAD_LETTER)

            stmt = stmt.limit(limit).offset(offset)

            result = await session.execute(stmt)
            jobs = result.scalars().all()

            return [_job_to_response(job) for job in jobs]

    async def cancel_job(self, job_id: UUID) -> dict:
        """Cancel a job if it is in a cancellable state.

        Cancellable states: PENDING, SCHEDULED, QUEUED.
        Non-cancellable states: RUNNING, COMPLETED, FAILED, DEAD_LETTER.
        Already CANCELLED: returns success idempotently.

        Steps:
            1. Fetch job from PostgreSQL by ID.
            2. If not found: return 404.
            3. If already CANCELLED: return 200 (idempotent).
            4. If in non-cancellable state: return 409.
            5. Apply CANCELLED transition, persist to PG, remove from Redis.

        Args:
            job_id: The UUID of the job to cancel.

        Returns:
            Dict with success/error info and status_code.
        """
        async with self._session_factory() as session:
            async with session.begin():
                stmt = select(Job).where(Job.id == job_id)
                result = await session.execute(stmt)
                job = result.scalar_one_or_none()

                if job is None:
                    return {"error": "not_found", "status_code": 404}

                # Idempotent: already cancelled
                if job.status == JobStatus.CANCELLED:
                    return {"success": True, "status_code": 200}

                # Non-cancellable states
                non_cancellable = {
                    JobStatus.RUNNING,
                    JobStatus.COMPLETED,
                    JobStatus.FAILED,
                    JobStatus.DEAD_LETTER,
                }
                if job.status in non_cancellable:
                    return {"error": "not_cancellable", "status_code": 409}

                # Capture current status before transition for Redis cleanup
                previous_status = job.status

                # Apply state machine transition to CANCELLED
                apply_transition(job, JobStatus.CANCELLED)

            # Session committed here - PG persisted before Redis operations

        # Remove from Redis based on previous status
        if previous_status == JobStatus.QUEUED:
            await self._priority_queue.remove(job_id)
        elif previous_status == JobStatus.SCHEDULED:
            await self._redis.zrem(SCHEDULED_QUEUE_KEY, str(job_id))
        # PENDING: no Redis removal needed

        return {"success": True, "status_code": 200}

    async def get_job_history(self, job_id: UUID) -> List[dict]:
        """Get execution history for a job ordered by attempt number.

        Args:
            job_id: The UUID of the job to get history for.

        Returns:
            List of execution records ordered by attempt_number ascending.
        """
        async with self._session_factory() as session:
            stmt = (
                select(JobExecution)
                .where(JobExecution.job_id == job_id)
                .order_by(JobExecution.attempt_number.asc())
            )
            result = await session.execute(stmt)
            executions = result.scalars().all()

            return [
                {
                    "id": execution.id,
                    "job_id": execution.job_id,
                    "worker_id": execution.worker_id,
                    "attempt_number": execution.attempt_number,
                    "status": execution.status,
                    "started_at": execution.started_at,
                    "completed_at": execution.completed_at,
                    "duration_ms": execution.duration_ms,
                    "error": execution.error,
                }
                for execution in executions
            ]

    async def list_dlq(self, limit: int = 50, offset: int = 0) -> List[dict]:
        """List jobs in the dead-letter queue with pagination.

        Queries PostgreSQL for jobs with DEAD_LETTER status, ordered by
        updated_at descending (most recently dead-lettered first).

        Args:
            limit: Maximum number of results (default 50, capped at 100).
            offset: Number of results to skip (default 0).

        Returns:
            List of DLQ job dicts with id, job_type, payload, error,
            retry_count, created_at, and updated_at.
        """
        # Cap limit at 100, enforce minimum of 1
        limit = min(limit, 100)
        if limit < 1:
            limit = 1
        if offset < 0:
            offset = 0

        async with self._session_factory() as session:
            stmt = (
                select(Job)
                .where(Job.status == JobStatus.DEAD_LETTER)
                .order_by(Job.updated_at.desc())
                .limit(limit)
                .offset(offset)
            )
            result = await session.execute(stmt)
            jobs = result.scalars().all()

            return [
                {
                    "id": job.id,
                    "job_type": job.job_type,
                    "payload": job.payload,
                    "error": job.error,
                    "retry_count": job.retry_count,
                    "created_at": job.created_at,
                    "updated_at": job.updated_at,
                }
                for job in jobs
            ]

    async def retry_dlq_job(self, job_id: UUID) -> dict:
        """Retry a dead-lettered job: reset retry_count, re-queue with original priority.

        Steps:
            1. Fetch job from PostgreSQL by ID.
            2. If not found: return 404 error.
            3. If not in DEAD_LETTER status: return 409 error.
            4. Apply DEAD_LETTER→QUEUED transition via state machine.
            5. Reset retry_count to 0, clear error field.
            6. Persist to PostgreSQL before Redis operations.
            7. Remove from DLQ sorted set in Redis.
            8. Add to priority queue with original priority.

        Args:
            job_id: The UUID of the dead-lettered job to retry.

        Returns:
            Dict with job info on success, or error info with status_code.
        """
        async with self._session_factory() as session:
            async with session.begin():
                stmt = select(Job).where(Job.id == job_id)
                result = await session.execute(stmt)
                job = result.scalar_one_or_none()

                if job is None:
                    return {"error": "not_found", "status_code": 404}

                if job.status != JobStatus.DEAD_LETTER:
                    return {"error": "not_retryable", "status_code": 409}

                # Apply state machine transition DEAD_LETTER → QUEUED
                apply_transition(job, JobStatus.QUEUED)

                # Reset retry_count and clear error (preserve payload and history)
                job.retry_count = 0
                job.error = None

                # Capture values needed for Redis operations
                job_id_val = job.id
                job_priority = job.priority
                job_created_at = job.created_at
                job_type_val = job.job_type
                job_payload = job.payload

            # Session committed here - PG persisted before Redis operations

        # Remove from DLQ sorted set in Redis
        await self._redis.zrem(DLQ_KEY, str(job_id_val))

        # Add to priority queue with original priority
        await self._priority_queue.enqueue(
            job_id=job_id_val,
            priority=job_priority,
            enqueued_at=job_created_at,
        )

        return {
            "id": job_id_val,
            "status": JobStatus.QUEUED.value,
            "job_type": job_type_val,
            "priority": job_priority,
            "payload": job_payload,
            "retry_count": 0,
            "status_code": 200,
        }


    async def list_workers(self) -> List[dict]:
        """List all workers with their current status.

        Returns each worker's ID, status, current job (if any),
        last heartbeat time, and jobs completed count.

        Returns:
            List of worker info dicts ordered by started_at descending.
        """
        async with self._session_factory() as session:
            stmt = select(Worker).order_by(Worker.started_at.desc())
            result = await session.execute(stmt)
            workers = result.scalars().all()

            return [
                {
                    "id": worker.id,
                    "status": worker.status.value,
                    "current_job_id": worker.current_job_id,
                    "last_heartbeat": worker.last_heartbeat,
                    "jobs_completed": worker.jobs_completed,
                }
                for worker in workers
            ]

    async def get_metrics(self) -> dict:
        """Get current system metrics.

        Queries queue depth from Redis, active workers from PostgreSQL,
        and DLQ size from Redis. Throughput and latency are provided by
        the MetricsCollector (placeholder zeros if not available).

        Returns:
            Dict with queue_depth, active_workers, jobs_per_second,
            latency_p50_ms, latency_p95_ms, and dlq_size.
        """
        # Get queue depth from priority queue
        queue_depth = await self._priority_queue.depth()

        # Get active worker count from PostgreSQL
        async with self._session_factory() as session:
            stmt = select(Worker).where(Worker.status == WorkerStatus.ACTIVE)
            result = await session.execute(stmt)
            active_workers = len(result.scalars().all())

        # Get DLQ size from Redis
        dlq_size = await self._redis.zcard(DLQ_KEY)

        return {
            "queue_depth": queue_depth,
            "active_workers": active_workers,
            "jobs_per_second": 0.0,
            "latency_p50_ms": 0.0,
            "latency_p95_ms": 0.0,
            "dlq_size": dlq_size,
        }


def _job_to_response(job: Job) -> dict:
    """Convert a Job model instance to a response dictionary.

    Args:
        job: The SQLAlchemy Job model instance.

    Returns:
        Dict containing all job metadata fields.
    """
    return {
        "id": job.id,
        "job_type": job.job_type,
        "status": job.status.value,
        "priority": job.priority,
        "payload": job.payload,
        "execute_at": job.execute_at,
        "timeout_seconds": job.timeout_seconds,
        "max_retries": job.max_retries,
        "retry_count": job.retry_count,
        "retry_backoff_base": job.retry_backoff_base,
        "worker_id": job.worker_id,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
        "result": job.result,
        "error": job.error,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
    }
