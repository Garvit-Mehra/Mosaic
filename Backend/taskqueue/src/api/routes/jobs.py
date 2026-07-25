"""Job management API routes.

Provides endpoints for job submission, status queries, history,
cancellation, and listing.
"""

from datetime import datetime
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from src.api.dependencies import get_coordinator, require_redis
from src.core.coordinator import JobCoordinator
from src.core.validators import ValidationError
from src.models.enums import JobStatus

router = APIRouter(prefix="/jobs", tags=["jobs"])


class JobSubmitRequest(BaseModel):
    """Request body for job submission."""

    job_type: str
    payload: dict
    priority: int = 0
    execute_at: Optional[datetime] = None
    timeout_seconds: int = 300
    max_retries: int = 3
    retry_backoff_base: float = 2.0


@router.post("", status_code=201)
async def submit_job(
    request: JobSubmitRequest,
    coordinator: JobCoordinator = Depends(get_coordinator),
    _redis_check: None = Depends(require_redis),
):
    """Submit a new job for processing.

    Creates a job in PostgreSQL and routes it to the priority queue (immediate)
    or schedule set (future execution).

    Args:
        request: Job submission parameters.
        coordinator: Injected JobCoordinator instance.

    Returns:
        Dict with job id, status, job_type, priority, and created_at.

    Raises:
        HTTPException: 400 for validation errors, 413 for oversized payloads.
    """
    try:
        result = await coordinator.submit_job(
            job_type=request.job_type,
            payload=request.payload,
            priority=request.priority,
            execute_at=request.execute_at,
            timeout_seconds=request.timeout_seconds,
            max_retries=request.max_retries,
            retry_backoff_base=request.retry_backoff_base,
        )
    except ValidationError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)

    return result


@router.get("")
async def list_jobs(
    status: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    coordinator: JobCoordinator = Depends(get_coordinator),
):
    """List jobs with optional status filter and pagination.

    Args:
        status: Optional job status filter (e.g. "QUEUED", "RUNNING").
        limit: Maximum number of results (1-100, default 50).
        offset: Number of results to skip (default 0).
        coordinator: Injected JobCoordinator instance.

    Returns:
        List of job response dicts.

    Raises:
        HTTPException: 400 if status filter value is invalid.
    """
    # Validate status filter if provided
    if status is not None:
        valid_statuses = [s.value for s in JobStatus]
        if status not in valid_statuses:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Invalid status filter '{status}'. "
                    f"Allowed values: {valid_statuses}"
                ),
            )

    return await coordinator.list_jobs(status=status, limit=limit, offset=offset)


@router.get("/{job_id}")
async def get_job(
    job_id: UUID,
    coordinator: JobCoordinator = Depends(get_coordinator),
):
    """Get job metadata by ID.

    Args:
        job_id: The UUID of the job to retrieve.
        coordinator: Injected JobCoordinator instance.

    Returns:
        Dict with complete job metadata.

    Raises:
        HTTPException: 404 if job not found.
    """
    result = await coordinator.get_job(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return result


@router.get("/{job_id}/history")
async def get_job_history(
    job_id: UUID,
    coordinator: JobCoordinator = Depends(get_coordinator),
):
    """Get execution history for a job.

    Returns all execution attempts ordered by attempt_number ascending.

    Args:
        job_id: The UUID of the job.
        coordinator: Injected JobCoordinator instance.

    Returns:
        List of execution attempt records.

    Raises:
        HTTPException: 404 if job not found.
    """
    # Verify job exists first
    job = await coordinator.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    return await coordinator.get_job_history(job_id)


@router.delete("/{job_id}")
async def cancel_job(
    job_id: UUID,
    coordinator: JobCoordinator = Depends(get_coordinator),
    _redis_check: None = Depends(require_redis),
):
    """Cancel a job by ID.

    Cancels jobs in PENDING, SCHEDULED, or QUEUED status.
    Returns 200 for successful cancellation or if already cancelled (idempotent).
    Returns 404 if the job does not exist.
    Returns 409 if the job is not in a cancellable state.

    Args:
        job_id: The UUID of the job to cancel.
        coordinator: Injected JobCoordinator instance.

    Returns:
        Dict with success message and job_id.

    Raises:
        HTTPException: 404 if job not found, 409 if not cancellable.
    """
    result = await coordinator.cancel_job(job_id)

    if result.get("error") == "not_found":
        raise HTTPException(status_code=404, detail="Job not found")

    if result.get("error") == "not_cancellable":
        raise HTTPException(
            status_code=409,
            detail="Job is not cancellable in its current state",
        )

    return {"message": "Job cancelled successfully", "job_id": str(job_id)}
