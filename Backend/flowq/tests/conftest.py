"""Global test configuration and fixtures.

Ensures the PostgreSQL health middleware doesn't block tests that use
create_app() when PG isn't available in the test environment.
Also ensures the Redis health check dependency doesn't block tests
when Redis isn't available.
"""

import pytest
from unittest.mock import AsyncMock, patch
from hypothesis import settings as hypothesis_settings

# Register and load a "fast" profile to reduce max_examples globally for faster test runs
hypothesis_settings.register_profile("fast", max_examples=50)
hypothesis_settings.load_profile("fast")

from src.api.middleware import reset_pg_health_cache, _pg_health_cache


@pytest.fixture(autouse=True)
def _reset_pg_health_for_tests():
    """Reset PG health cache and mark PG as healthy for all tests.

    This ensures that tests for other functionality are not blocked by the
    PG health middleware when PostgreSQL isn't running in the test environment.
    Tests that specifically test PG failure handling will override this by
    directly manipulating the cache.
    """
    reset_pg_health_cache()
    # Set the cache to healthy with a future timestamp so no real check is performed
    _pg_health_cache["healthy"] = True
    _pg_health_cache["last_check"] = 9999999999.0
    yield
    # Reset after test
    reset_pg_health_cache()


@pytest.fixture(autouse=True)
def _mock_redis_health_for_tests():
    """Mock Redis health check to return True for all tests.

    Tests that specifically test Redis failure handling will override
    this by patching check_redis_health themselves.
    """
    with patch("src.api.dependencies.check_redis_health", return_value=True):
        yield
