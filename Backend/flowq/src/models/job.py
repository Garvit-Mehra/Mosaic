"""SQLAlchemy model for the Job entity."""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, JSON, String, Text
from sqlalchemy import Enum as SAEnum
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from src.models.base import Base
from src.models.enums import JobStatus


class Job(Base):
    """Represents a job in the distributed job queue."""

    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    job_type: Mapped[str] = mapped_column(
        String(255), nullable=False, index=True
    )
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    priority: Mapped[int] = mapped_column(Integer, default=0, index=True)
    status: Mapped[JobStatus] = mapped_column(
        SAEnum(JobStatus, name="jobstatus", create_constraint=True),
        default=JobStatus.PENDING,
        index=True,
    )

    # Scheduling
    execute_at: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True, index=True
    )

    # Retry configuration
    max_retries: Mapped[int] = mapped_column(Integer, default=3)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    retry_backoff_base: Mapped[float] = mapped_column(Float, default=2.0)
    timeout_seconds: Mapped[int] = mapped_column(Integer, default=300)

    # Execution tracking
    worker_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True), nullable=True, index=True
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True
    )

    # Results
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    def __repr__(self) -> str:
        return f"<Job(id={self.id}, type={self.job_type}, status={self.status})>"
