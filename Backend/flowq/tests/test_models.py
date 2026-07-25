"""Unit tests for SQLAlchemy models."""

import uuid
from datetime import datetime

from src.models import Base, Job, JobExecution, JobStatus, Worker, WorkerStatus


class TestEnums:
    """Test enum definitions."""

    def test_job_status_values(self):
        """JobStatus enum has all expected values."""
        expected = {
            "PENDING", "SCHEDULED", "QUEUED", "RUNNING",
            "COMPLETED", "FAILED", "CANCELLED", "DEAD_LETTER",
        }
        actual = {s.value for s in JobStatus}
        assert actual == expected

    def test_worker_status_values(self):
        """WorkerStatus enum has all expected values."""
        expected = {"ACTIVE", "IDLE", "DEAD", "SHUTTING_DOWN"}
        actual = {s.value for s in WorkerStatus}
        assert actual == expected

    def test_job_status_is_str_enum(self):
        """JobStatus values are strings."""
        assert JobStatus.PENDING == "PENDING"
        assert isinstance(JobStatus.RUNNING, str)

    def test_worker_status_is_str_enum(self):
        """WorkerStatus values are strings."""
        assert WorkerStatus.ACTIVE == "ACTIVE"
        assert isinstance(WorkerStatus.IDLE, str)


class TestJobModel:
    """Test Job model structure."""

    def test_job_tablename(self):
        """Job model maps to 'jobs' table."""
        assert Job.__tablename__ == "jobs"

    def test_job_columns_exist(self):
        """Job model has all required columns."""
        columns = {c.name for c in Job.__table__.columns}
        expected = {
            "id", "job_type", "payload", "priority", "status",
            "execute_at", "max_retries", "retry_count", "retry_backoff_base",
            "timeout_seconds", "worker_id", "started_at", "completed_at",
            "result", "error", "created_at", "updated_at",
        }
        assert expected.issubset(columns)

    def test_job_primary_key(self):
        """Job has UUID primary key."""
        pk_cols = [c for c in Job.__table__.columns if c.primary_key]
        assert len(pk_cols) == 1
        assert pk_cols[0].name == "id"

    def test_job_indexes(self):
        """Job model has expected indexes."""
        indexed_columns = set()
        for idx in Job.__table__.indexes:
            for col in idx.columns:
                indexed_columns.add(col.name)
        expected_indexed = {"job_type", "priority", "status", "execute_at", "worker_id"}
        assert expected_indexed.issubset(indexed_columns)

    def test_job_in_metadata(self):
        """Job table is registered in Base metadata."""
        assert "jobs" in Base.metadata.tables


class TestWorkerModel:
    """Test Worker model structure."""

    def test_worker_tablename(self):
        """Worker model maps to 'workers' table."""
        assert Worker.__tablename__ == "workers"

    def test_worker_columns_exist(self):
        """Worker model has all required columns."""
        columns = {c.name for c in Worker.__table__.columns}
        expected = {
            "id", "hostname", "pid", "status", "current_job_id",
            "last_heartbeat", "started_at", "jobs_completed", "jobs_failed",
        }
        assert expected.issubset(columns)

    def test_worker_primary_key(self):
        """Worker has UUID primary key."""
        pk_cols = [c for c in Worker.__table__.columns if c.primary_key]
        assert len(pk_cols) == 1
        assert pk_cols[0].name == "id"

    def test_worker_in_metadata(self):
        """Worker table is registered in Base metadata."""
        assert "workers" in Base.metadata.tables


class TestJobExecutionModel:
    """Test JobExecution model structure."""

    def test_job_execution_tablename(self):
        """JobExecution model maps to 'job_executions' table."""
        assert JobExecution.__tablename__ == "job_executions"

    def test_job_execution_columns_exist(self):
        """JobExecution model has all required columns."""
        columns = {c.name for c in JobExecution.__table__.columns}
        expected = {
            "id", "job_id", "worker_id", "attempt_number",
            "status", "started_at", "completed_at", "duration_ms", "error",
        }
        assert expected.issubset(columns)

    def test_job_execution_primary_key(self):
        """JobExecution has UUID primary key."""
        pk_cols = [c for c in JobExecution.__table__.columns if c.primary_key]
        assert len(pk_cols) == 1
        assert pk_cols[0].name == "id"

    def test_job_execution_job_id_indexed(self):
        """JobExecution has job_id indexed for query performance."""
        indexed_columns = set()
        for idx in JobExecution.__table__.indexes:
            for col in idx.columns:
                indexed_columns.add(col.name)
        assert "job_id" in indexed_columns

    def test_job_execution_in_metadata(self):
        """JobExecution table is registered in Base metadata."""
        assert "job_executions" in Base.metadata.tables
