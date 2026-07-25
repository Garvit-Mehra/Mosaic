"""initial_models

Revision ID: ab7a391a5777
Revises: 
Create Date: 2026-07-24 23:31:19.145480

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID


# revision identifiers, used by Alembic.
revision: str = 'ab7a391a5777'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create initial tables: jobs, workers, job_executions."""
    # Create enums
    jobstatus_enum = sa.Enum(
        'PENDING', 'SCHEDULED', 'QUEUED', 'RUNNING',
        'COMPLETED', 'FAILED', 'CANCELLED', 'DEAD_LETTER',
        name='jobstatus'
    )
    workerstatus_enum = sa.Enum(
        'ACTIVE', 'IDLE', 'DEAD', 'SHUTTING_DOWN',
        name='workerstatus'
    )

    # Create jobs table
    op.create_table(
        'jobs',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('job_type', sa.String(255), nullable=False),
        sa.Column('payload', sa.JSON(), nullable=False),
        sa.Column('priority', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('status', jobstatus_enum, nullable=False, server_default='PENDING'),
        sa.Column('execute_at', sa.DateTime(), nullable=True),
        sa.Column('max_retries', sa.Integer(), nullable=False, server_default='3'),
        sa.Column('retry_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('retry_backoff_base', sa.Float(), nullable=False, server_default='2.0'),
        sa.Column('timeout_seconds', sa.Integer(), nullable=False, server_default='300'),
        sa.Column('worker_id', UUID(as_uuid=True), nullable=True),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('result', sa.JSON(), nullable=True),
        sa.Column('error', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
    )
    op.create_index('ix_jobs_job_type', 'jobs', ['job_type'])
    op.create_index('ix_jobs_priority', 'jobs', ['priority'])
    op.create_index('ix_jobs_status', 'jobs', ['status'])
    op.create_index('ix_jobs_execute_at', 'jobs', ['execute_at'])
    op.create_index('ix_jobs_worker_id', 'jobs', ['worker_id'])

    # Create workers table
    op.create_table(
        'workers',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('hostname', sa.String(255), nullable=False),
        sa.Column('pid', sa.Integer(), nullable=False),
        sa.Column('status', workerstatus_enum, nullable=False, server_default='IDLE'),
        sa.Column('current_job_id', UUID(as_uuid=True), nullable=True),
        sa.Column('last_heartbeat', sa.DateTime(), nullable=False),
        sa.Column('started_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.Column('jobs_completed', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('jobs_failed', sa.Integer(), nullable=False, server_default='0'),
    )

    # Create job_executions table
    op.create_table(
        'job_executions',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('job_id', UUID(as_uuid=True), nullable=False),
        sa.Column('worker_id', UUID(as_uuid=True), nullable=False),
        sa.Column('attempt_number', sa.Integer(), nullable=False),
        sa.Column('status', sa.String(50), nullable=False),
        sa.Column('started_at', sa.DateTime(), nullable=False),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('duration_ms', sa.Integer(), nullable=True),
        sa.Column('error', sa.Text(), nullable=True),
    )
    op.create_index('ix_job_executions_job_id', 'job_executions', ['job_id'])


def downgrade() -> None:
    """Drop all tables and enums."""
    op.drop_table('job_executions')
    op.drop_table('workers')
    op.drop_table('jobs')

    # Drop enums
    sa.Enum(name='jobstatus').drop(op.get_bind(), checkfirst=True)
    sa.Enum(name='workerstatus').drop(op.get_bind(), checkfirst=True)
