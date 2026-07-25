"""Dead-Letter Queue API routes.

Provides endpoints for listing DLQ contents and retrying dead-lettered jobs.
"""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.dependencies import get_coordinator
from src.core.coordinator import JobCoordinator

router = APIRouter(prefix="/dlq", tags=["dlq"])


@router.get("")
async def list_dlq(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    coordinator: JobCoordinator = Depends(get_coordinator),
):
    """List jobs in the dead-letter queue with pagination.

    Returns dead-lettered jobs ordered by most recently dead-lettered first.
    DLQ jobs do not appear in general job listings unless explicitly filtered
    by DEAD_LETTER status.

    Args:
        limit: Maximum number of results (1-100, default 50).
        offset: Number of results to skip (default 0).
        coordinator: Injected JobCoordinator instance.

    Returns:
        Dict with list of DLQ jobs and pagination metadata.
    """
    jobs = await coordinator.list_dlq(limit=limit, offset=offset)
    return {
        "jobs": jobs,
        "limit": limit,
        "offset": offset,
        "count": len(jobs),
    }


@router.post("/{job_id}/retry")
async def retry_dlq_job(
    job_id: UUID,
    coordinator: JobCoordinator = Depends(get_coordinator),
):
    """Retry a dead-lettered job.

    Resets the job's retry_count to 0, transitions it from DEAD_LETTER to QUEUED,
    and adds it back to the priority queue with its original priority.
    Preserves the original payload and execution history.

    Args:
        job_id: The UUID of the dead-lettered job to retry.
        coordinator: Injected JobCoordinator instance.

    Returns:
        Dict with the retried job's information.

    Raises:
        HTTPException: 404 if job not found, 409 if not in DEAD_LETTER state.
    """
    result = await coordinator.retry_dlq_job(job_id)

    if result.get("error") == "not_found":
        raise HTTPException(status_code=404, detail="Job not found")

    if result.get("error") == "not_retryable":
        raise HTTPException(
            status_code=409,
            detail="Job is not in dead-letter state",
        )

    # Remove internal status_code from response
    result.pop("status_code", None)
    return result
