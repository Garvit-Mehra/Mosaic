"""
Mosaic <-> TaskQueue Integration Bridge

This module connects Mosaic's agent system to the distributed job queue.
Long-running operations (RAG processing, MCP tool calls, heavy computations)
are submitted as background jobs instead of blocking the chat response.

Usage from Mosaic:
    from taskqueue.src.mosaic_bridge import submit_background_job, get_job_result

    # Submit a long-running task
    job_id = await submit_background_job(
        job_type="mcp_tool_call",
        payload={"server": "database", "tool": "execute_sql", "args": {...}},
        user_id="admin",
        priority=5,
    )

    # Check result later
    result = await get_job_result(job_id)
"""

import logging
from typing import Any, Dict, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


async def get_coordinator():
    """Get the TaskQueue coordinator from the running app state."""
    from taskqueue.src.database import async_session_factory
    from taskqueue.src.redis_client import get_redis_client
    from taskqueue.src.core.priority_queue import RedisPriorityQueue
    from taskqueue.src.core.coordinator import JobCoordinator

    redis_client = get_redis_client()
    priority_queue = RedisPriorityQueue(redis_client=redis_client)
    return JobCoordinator(
        session_factory=async_session_factory,
        redis_client=redis_client,
        priority_queue=priority_queue,
    )


async def submit_background_job(
    job_type: str,
    payload: Dict[str, Any],
    user_id: str,
    priority: int = 5,
    timeout_seconds: int = 300,
    max_retries: int = 2,
    scheduled_at: Optional[datetime] = None,
) -> str:
    """
    Submit a background job to the task queue.

    Args:
        job_type: Type of job (e.g., "mcp_tool_call", "rag_process", "web_scrape")
        payload: Job-specific data
        user_id: Who submitted the job
        priority: 1 (lowest) to 10000 (highest), default 5
        timeout_seconds: Max execution time
        max_retries: How many times to retry on failure
        scheduled_at: Optional future execution time

    Returns:
        job_id (str) for tracking
    """
    coordinator = await get_coordinator()

    job_data = {
        "job_type": job_type,
        "payload": payload,
        "priority": priority,
        "timeout_seconds": timeout_seconds,
        "max_retries": max_retries,
        "metadata": {"user_id": user_id, "source": "mosaic"},
    }

    if scheduled_at:
        job_data["scheduled_at"] = scheduled_at.isoformat()

    job = await coordinator.submit_job(job_data)
    logger.info(f"Submitted background job: type={job_type} id={job.id} user={user_id}")
    return str(job.id)


async def get_job_result(job_id: str) -> Dict[str, Any]:
    """
    Get the current status and result of a background job.

    Returns:
        {
            "id": "...",
            "status": "completed|running|queued|failed|dead_letter",
            "result": {...} or None,
            "error": "..." or None,
            "progress": 0-100,
        }
    """
    coordinator = await get_coordinator()
    job = await coordinator.get_job(job_id)

    if not job:
        return {"id": job_id, "status": "not_found", "result": None, "error": None}

    return {
        "id": str(job.id),
        "status": job.status,
        "result": job.result if hasattr(job, "result") else None,
        "error": job.error if hasattr(job, "error") else None,
        "created_at": str(job.created_at) if hasattr(job, "created_at") else None,
        "completed_at": str(job.completed_at) if hasattr(job, "completed_at") else None,
    }


async def cancel_job(job_id: str) -> bool:
    """Cancel a pending/queued job."""
    coordinator = await get_coordinator()
    try:
        await coordinator.cancel_job(job_id)
        return True
    except Exception as e:
        logger.error(f"Failed to cancel job {job_id}: {e}")
        return False


# ─────────────────────────────────────────────────────────────
# Pre-defined job types for Mosaic
# ─────────────────────────────────────────────────────────────

JOB_TYPES = {
    "mcp_tool_call": "Execute an MCP server tool call in the background",
    "rag_process": "Process and index a document for RAG",
    "web_scrape": "Fetch and process content from a URL",
    "batch_chat": "Process multiple chat messages sequentially",
}
