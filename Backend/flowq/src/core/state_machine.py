"""Job state machine with transition validation.

Defines valid state transitions for the job lifecycle and provides
functions to validate and apply transitions. This module is pure logic
with no database dependencies.
"""

from datetime import datetime

from src.models.enums import JobStatus


class InvalidTransitionError(Exception):
    """Raised when an invalid state transition is attempted."""

    def __init__(self, current_status: JobStatus, target_status: JobStatus, reason: str = ""):
        self.current_status = current_status
        self.target_status = target_status
        if reason:
            message = (
                f"Invalid transition from {current_status.value} to {target_status.value}: {reason}"
            )
        else:
            message = (
                f"Invalid transition from {current_status.value} to {target_status.value}"
            )
        super().__init__(message)


# Terminal states reject ALL outgoing transitions (except DLQ retry, handled separately)
TERMINAL_STATES: frozenset[JobStatus] = frozenset({
    JobStatus.COMPLETED,
    JobStatus.CANCELLED,
})

# Valid transitions map: current_status -> set of allowed target statuses
VALID_TRANSITIONS: dict[JobStatus, set[JobStatus]] = {
    JobStatus.PENDING: {JobStatus.QUEUED, JobStatus.SCHEDULED, JobStatus.CANCELLED},
    JobStatus.SCHEDULED: {JobStatus.QUEUED, JobStatus.CANCELLED},
    JobStatus.QUEUED: {JobStatus.RUNNING, JobStatus.CANCELLED},
    JobStatus.RUNNING: {JobStatus.COMPLETED, JobStatus.FAILED},
    JobStatus.FAILED: {JobStatus.QUEUED, JobStatus.DEAD_LETTER},
    # Terminal states have no valid outgoing transitions
    JobStatus.COMPLETED: set(),
    JobStatus.CANCELLED: set(),
    JobStatus.DEAD_LETTER: {JobStatus.QUEUED},
}


def validate_transition(current_status: JobStatus, target_status: JobStatus) -> bool:
    """Check whether a state transition is valid.

    Args:
        current_status: The current status of the job.
        target_status: The desired target status.

    Returns:
        True if the transition is allowed, False otherwise.
    """
    allowed = VALID_TRANSITIONS.get(current_status, set())
    return target_status in allowed


def apply_transition(job, target_status: JobStatus) -> None:
    """Apply a state transition to a job, updating status and updated_at.

    Validates the transition before applying. If the transition is invalid,
    raises InvalidTransitionError and leaves the job unchanged.

    Args:
        job: A job object with `status` (JobStatus) and `updated_at` (datetime) attributes.
        target_status: The desired target status.

    Raises:
        InvalidTransitionError: If the transition is not allowed.
    """
    current_status = job.status

    if current_status in TERMINAL_STATES:
        raise InvalidTransitionError(
            current_status,
            target_status,
            reason=f"no transitions are allowed from terminal state {current_status.value}",
        )

    if not validate_transition(current_status, target_status):
        raise InvalidTransitionError(current_status, target_status)

    # Apply the transition atomically (status + timestamp)
    job.status = target_status
    job.updated_at = datetime.utcnow()
