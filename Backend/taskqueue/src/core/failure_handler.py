"""Job failure handling with retry logic and dead-letter queue routing.

Implements the failure path for jobs that error during execution:
- If retries remain: increment retry_count, calculate backoff, add to retry queue
- If retries exhausted: transition to DEAD_LETTER, add to DLQ sorted set
- Always records execution history and clears worker assignment
"""

from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from src.core.backoff import calculate_backoff
from src.core.state_machine import apply_transition
from src.models.enums import JobStatus
from src.models.job_execution import JobExecution

# Redis keys for retry and dead-letter queues
RETRY_QUEUE_KEY = "queue:retry"
DLQ_KEY = "queue:dlq"


async def handle_job_failure(
    job,
    error: str,
    session: AsyncSession,
    redis_client,
) -> None:
    """Handle a failed job: retry with backoff or move to dead-letter queue.

    Implements Algorithm 3 from the design document. The job transitions
    through RUNNING → FAILED, then either FAILED → QUEUED (retry) or
    FAILED → DEAD_LETTER (exhausted).

    Args:
        job: SQLAlchemy Job model instance (must be in RUNNING status).
        error: Descriptive error string explaining the failure.
        session: Active async SQLAlchemy session for persistence.
        redis_client: Async Redis client for queue operations.
    """
    now = datetime.utcnow()

    # Capture execution details before state changes
    worker_id = job.worker_id
    started_at = job.started_at
    attempt_number = job.retry_count + 1

    # Transition RUNNING → FAILED
    apply_transition(job, JobStatus.FAILED)

    new_retry_count = job.retry_count + 1

    if new_retry_count <= job.max_retries:
        # Retries remaining: transition FAILED → QUEUED and add to retry queue
        apply_transition(job, JobStatus.QUEUED)

        # Calculate exponential backoff delay
        backoff_seconds = calculate_backoff(
            retry_count=new_retry_count,
            base=job.retry_backoff_base,
        )
        next_retry_at = now + timedelta(seconds=backoff_seconds)
        retry_score = int(next_retry_at.timestamp() * 1000)

        # Update job fields for retry
        job.retry_count = new_retry_count
        job.error = error
        job.worker_id = None
        job.started_at = None

        # Add to retry queue in Redis
        await redis_client.zadd(RETRY_QUEUE_KEY, {str(job.id): retry_score})

    else:
        # Max retries exhausted: transition FAILED → DEAD_LETTER
        apply_transition(job, JobStatus.DEAD_LETTER)

        # Update job fields for DLQ
        job.retry_count = new_retry_count
        job.error = error
        job.worker_id = None

        # Add to DLQ sorted set in Redis
        failed_at_score = int(now.timestamp() * 1000)
        await redis_client.zadd(DLQ_KEY, {str(job.id): failed_at_score})

    # Record execution history
    await _record_execution(
        session=session,
        job_id=job.id,
        worker_id=worker_id,
        attempt_number=attempt_number,
        status="FAILED",
        started_at=started_at or now,
        completed_at=now,
        error=error,
    )

    # Persist all job changes
    session.add(job)
    await session.flush()


async def _record_execution(
    session: AsyncSession,
    job_id: UUID,
    worker_id: UUID,
    attempt_number: int,
    status: str,
    started_at: datetime,
    completed_at: datetime,
    error: Optional[str] = None,
) -> None:
    """Record a job execution attempt in the execution history table.

    Args:
        session: Active async SQLAlchemy session.
        job_id: UUID of the job that was executed.
        worker_id: UUID of the worker that executed the job.
        attempt_number: The retry attempt number (1-based).
        status: Execution result status (e.g. "FAILED", "COMPLETED").
        started_at: When execution began.
        completed_at: When execution ended.
        error: Optional error message if the execution failed.
    """
    duration_ms = int((completed_at - started_at).total_seconds() * 1000)

    execution = JobExecution(
        job_id=job_id,
        worker_id=worker_id,
        attempt_number=attempt_number,
        status=status,
        started_at=started_at,
        completed_at=completed_at,
        duration_ms=duration_ms,
        error=error,
    )
    session.add(execution)
