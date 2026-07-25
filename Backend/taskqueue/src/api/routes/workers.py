"""Worker status API routes.

Provides an endpoint for listing all workers with their current status,
current job assignment, last heartbeat, and jobs completed count.
"""

from fastapi import APIRouter, Depends

from src.api.dependencies import get_coordinator
from src.core.coordinator import JobCoordinator

router = APIRouter(prefix="/workers", tags=["workers"])


@router.get("")
async def list_workers(
    coordinator: JobCoordinator = Depends(get_coordinator),
):
    """List all workers with their current status.

    Returns each worker's ID, status, current job (if any),
    last heartbeat time, and jobs completed count.

    Args:
        coordinator: Injected JobCoordinator instance.

    Returns:
        List of worker info dicts.
    """
    workers = await coordinator.list_workers()
    return workers
