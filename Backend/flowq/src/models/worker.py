"""SQLAlchemy model for the Worker entity."""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Integer, String
from sqlalchemy import Enum as SAEnum
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from src.models.base import Base
from src.models.enums import WorkerStatus


class Worker(Base):
    """Represents a worker process in the distributed job queue."""

    __tablename__ = "workers"

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    hostname: Mapped[str] = mapped_column(String(255), nullable=False)
    pid: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[WorkerStatus] = mapped_column(
        SAEnum(WorkerStatus, name="workerstatus", create_constraint=True),
        default=WorkerStatus.IDLE,
    )
    current_job_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True), nullable=True
    )
    last_heartbeat: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    started_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow
    )
    jobs_completed: Mapped[int] = mapped_column(Integer, default=0)
    jobs_failed: Mapped[int] = mapped_column(Integer, default=0)

    def __repr__(self) -> str:
        return f"<Worker(id={self.id}, hostname={self.hostname}, status={self.status})>"
