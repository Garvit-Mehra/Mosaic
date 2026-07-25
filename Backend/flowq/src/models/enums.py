"""Job and Worker status enums for the distributed job queue."""

from enum import Enum


class JobStatus(str, Enum):
    """Possible states in the job lifecycle."""

    PENDING = "PENDING"
    SCHEDULED = "SCHEDULED"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    DEAD_LETTER = "DEAD_LETTER"


class WorkerStatus(str, Enum):
    """Possible states of a worker process."""

    ACTIVE = "ACTIVE"
    IDLE = "IDLE"
    DEAD = "DEAD"
    SHUTTING_DOWN = "SHUTTING_DOWN"
