"""Property-based tests for job status filter accuracy and DLQ isolation.

**Validates: Requirements 2.3, 14.4**

Uses Hypothesis to generate random sets of jobs with various statuses and verify:
- Property 15: Filtered job listings return ONLY jobs with the specified status (Req 2.3)
- Property 16: DLQ jobs appear in the DLQ endpoint but NOT in general unfiltered listings (Req 14.4)
"""

import uuid
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.core.coordinator import JobCoordinator
from src.models.base import Base
from src.models.enums import JobStatus
from src.models.job import Job


# --- Strategies ---

# All valid job statuses
job_status_strategy = st.sampled_from(list(JobStatus))

# Non-DLQ statuses (for general listing)
non_dlq_status_strategy = st.sampled_from(
    [s for s in JobStatus if s != JobStatus.DEAD_LETTER]
)

# Valid job types
job_type_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="_-"),
    min_size=1,
    max_size=30,
)

# Valid priorities
priority_strategy = st.integers(min_value=0, max_value=10000)


# --- Job generation strategy ---

@st.composite
def job_set_strategy(draw, min_size=3, max_size=15):
    """Generate a list of job specs with random statuses for insertion into the DB."""
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    jobs = []
    for i in range(size):
        status = draw(job_status_strategy)
        job_type = draw(job_type_strategy)
        priority = draw(priority_strategy)
        jobs.append({
            "status": status,
            "job_type": job_type,
            "priority": priority,
        })
    return jobs


@st.composite
def job_set_with_dlq_strategy(draw):
    """Generate a job set that includes at least one DLQ job and one non-DLQ job."""
    # Generate some non-DLQ jobs
    non_dlq_count = draw(st.integers(min_value=1, max_value=8))
    dlq_count = draw(st.integers(min_value=1, max_value=5))

    jobs = []
    for _ in range(non_dlq_count):
        status = draw(non_dlq_status_strategy)
        job_type = draw(job_type_strategy)
        priority = draw(priority_strategy)
        jobs.append({
            "status": status,
            "job_type": job_type,
            "priority": priority,
        })

    for _ in range(dlq_count):
        job_type = draw(job_type_strategy)
        priority = draw(priority_strategy)
        jobs.append({
            "status": JobStatus.DEAD_LETTER,
            "job_type": job_type,
            "priority": priority,
        })

    return jobs


# --- Helper to create coordinator with seeded jobs ---


async def _create_coordinator_with_jobs(job_specs):
    """Create a coordinator with a fresh in-memory DB and insert jobs with given statuses.

    Returns (coordinator, engine, inserted_jobs) where inserted_jobs is a list of
    dicts with 'id' and 'status' for each created job.
    """
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    mock_redis = AsyncMock()
    mock_priority_queue = AsyncMock()

    coordinator = JobCoordinator(
        session_factory=session_factory,
        redis_client=mock_redis,
        priority_queue=mock_priority_queue,
    )

    # Insert jobs directly into the database with desired statuses
    inserted_jobs = []
    async with session_factory() as session:
        async with session.begin():
            base_time = datetime.utcnow()
            for i, spec in enumerate(job_specs):
                job = Job(
                    id=uuid.uuid4(),
                    job_type=spec["job_type"] if spec["job_type"] else "test_type",
                    payload={"index": i},
                    priority=spec["priority"],
                    status=spec["status"],
                    max_retries=3,
                    retry_count=0,
                    retry_backoff_base=2.0,
                    timeout_seconds=300,
                    created_at=base_time - timedelta(seconds=i),
                    updated_at=base_time - timedelta(seconds=i),
                )
                if spec["status"] == JobStatus.DEAD_LETTER:
                    job.error = "Max retries exceeded"
                session.add(job)
                inserted_jobs.append({
                    "id": job.id,
                    "status": spec["status"],
                })

    return coordinator, engine, inserted_jobs


# --- Property Tests ---


class TestJobStatusFilterAccuracy:
    """Property 15: Job Status Filter Accuracy.

    **Validates: Requirements 2.3**

    For any valid status filter applied to list_jobs, every returned job SHALL have
    that exact status, and no jobs with that status SHALL be omitted (within pagination bounds).
    """

    @given(job_specs=job_set_strategy(min_size=3, max_size=15), filter_status=job_status_strategy)
    @settings(max_examples=100)
    async def test_filtered_listing_returns_only_matching_status(
        self, job_specs, filter_status
    ):
        """Filtering by any status returns ONLY jobs with that exact status.

        **Validates: Requirements 2.3**

        Property: For ALL valid status values, list_jobs(status=X) returns
        a set where every job has status == X.
        """
        coordinator, engine, inserted_jobs = await _create_coordinator_with_jobs(job_specs)

        try:
            # Apply the filter
            results = await coordinator.list_jobs(
                status=filter_status.value, limit=100, offset=0
            )

            # Every returned job must have the filtered status
            for job_dict in results:
                assert job_dict["status"] == filter_status.value, (
                    f"Expected all jobs to have status {filter_status.value}, "
                    f"but got {job_dict['status']}"
                )

            # Count how many inserted jobs have this status
            expected_count = sum(
                1 for j in inserted_jobs if j["status"] == filter_status
            )

            # The result count must match the expected count (within limit)
            assert len(results) == expected_count, (
                f"Expected {expected_count} jobs with status {filter_status.value}, "
                f"got {len(results)}"
            )
        finally:
            await engine.dispose()

    @given(job_specs=job_set_strategy(min_size=5, max_size=15), filter_status=job_status_strategy)
    @settings(max_examples=80)
    async def test_filtered_listing_completeness(
        self, job_specs, filter_status
    ):
        """No jobs with the specified status are omitted from the filtered result.

        **Validates: Requirements 2.3**

        Property: The set of job IDs returned by list_jobs(status=X) equals the set
        of all inserted job IDs that have status X (within pagination bounds).
        """
        coordinator, engine, inserted_jobs = await _create_coordinator_with_jobs(job_specs)

        try:
            results = await coordinator.list_jobs(
                status=filter_status.value, limit=100, offset=0
            )

            # Get expected job IDs
            expected_ids = {
                j["id"] for j in inserted_jobs if j["status"] == filter_status
            }
            result_ids = {r["id"] for r in results}

            assert result_ids == expected_ids, (
                f"Filter for {filter_status.value}: "
                f"missing IDs: {expected_ids - result_ids}, "
                f"extra IDs: {result_ids - expected_ids}"
            )
        finally:
            await engine.dispose()


class TestDLQIsolation:
    """Property 16: DLQ Isolation.

    **Validates: Requirements 14.4**

    For any job in the Dead_Letter_Queue, the job SHALL appear in the DLQ endpoint
    but SHALL NOT appear in general job listings (unless explicitly filtered by
    DEAD_LETTER status).
    """

    @given(job_specs=job_set_with_dlq_strategy())
    @settings(max_examples=80)
    async def test_dlq_jobs_appear_in_dlq_listing(self, job_specs):
        """All DEAD_LETTER jobs appear in the DLQ endpoint listing.

        **Validates: Requirements 14.4**

        Property: For ALL jobs with status DEAD_LETTER, they are present
        in the list_dlq() result set.
        """
        coordinator, engine, inserted_jobs = await _create_coordinator_with_jobs(job_specs)

        try:
            dlq_results = await coordinator.list_dlq(limit=100, offset=0)

            # Get all DLQ job IDs from inserted jobs
            expected_dlq_ids = {
                j["id"] for j in inserted_jobs if j["status"] == JobStatus.DEAD_LETTER
            }
            result_dlq_ids = {r["id"] for r in dlq_results}

            # All DLQ jobs must appear in the DLQ listing
            assert expected_dlq_ids == result_dlq_ids, (
                f"DLQ listing missing jobs: {expected_dlq_ids - result_dlq_ids}, "
                f"unexpected jobs: {result_dlq_ids - expected_dlq_ids}"
            )
        finally:
            await engine.dispose()

    @given(job_specs=job_set_with_dlq_strategy())
    @settings(max_examples=80)
    async def test_dlq_jobs_not_in_general_listing(self, job_specs):
        """DEAD_LETTER jobs do NOT appear in unfiltered general job listings.

        **Validates: Requirements 14.4**

        Property: For ALL jobs with status DEAD_LETTER, calling list_jobs()
        without a status filter SHALL NOT include them in the results.
        Note: Requirement 14.4 states DLQ jobs should not appear in general listings.
        """
        coordinator, engine, inserted_jobs = await _create_coordinator_with_jobs(job_specs)

        try:
            # Get general listing without any status filter
            general_results = await coordinator.list_jobs(limit=100, offset=0)

            # No DEAD_LETTER job should appear in general unfiltered listing
            dlq_ids = {
                j["id"] for j in inserted_jobs if j["status"] == JobStatus.DEAD_LETTER
            }
            general_result_ids = {r["id"] for r in general_results}

            dlq_in_general = dlq_ids & general_result_ids
            assert len(dlq_in_general) == 0, (
                f"DLQ jobs appeared in general listing: {dlq_in_general}. "
                f"Requirement 14.4 states DLQ jobs should not appear in general listings."
            )
        finally:
            await engine.dispose()

    @given(job_specs=job_set_with_dlq_strategy())
    @settings(max_examples=80)
    async def test_dlq_jobs_accessible_via_explicit_status_filter(self, job_specs):
        """DEAD_LETTER jobs ARE accessible when explicitly filtered by DEAD_LETTER status.

        **Validates: Requirements 14.4**

        Property: list_jobs(status="DEAD_LETTER") returns exactly the same jobs
        as list_dlq(), confirming DLQ jobs are accessible via explicit filter
        but isolated from general listings.
        """
        coordinator, engine, inserted_jobs = await _create_coordinator_with_jobs(job_specs)

        try:
            # Get DLQ jobs via status filter
            filtered_results = await coordinator.list_jobs(
                status=JobStatus.DEAD_LETTER.value, limit=100, offset=0
            )

            # Get DLQ jobs via dedicated DLQ endpoint
            dlq_results = await coordinator.list_dlq(limit=100, offset=0)

            # Both should return the same set of job IDs
            filtered_ids = {r["id"] for r in filtered_results}
            dlq_ids = {r["id"] for r in dlq_results}

            assert filtered_ids == dlq_ids, (
                f"Mismatch between status filter and DLQ endpoint: "
                f"only in filter: {filtered_ids - dlq_ids}, "
                f"only in DLQ: {dlq_ids - filtered_ids}"
            )
        finally:
            await engine.dispose()
