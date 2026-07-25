"""Basic tests to verify project scaffold."""

from src.config import Settings, settings
from src.main import app, create_app


def test_settings_defaults():
    """Verify default settings are properly configured."""
    s = Settings()
    assert s.app_name == "Distributed Job Queue"
    assert s.redis_url == "redis://localhost:6379/0"
    assert s.postgres_url == "postgresql+asyncpg://postgres:postgres@localhost:5432/jobqueue"
    assert s.worker_count == 4
    assert s.heartbeat_interval_seconds == 5
    assert s.heartbeat_ttl_seconds == 15
    assert s.payload_size_limit_bytes == 1_048_576
    assert s.max_priority == 10000
    assert s.default_max_retries == 3
    assert s.max_job_timeout_seconds == 86400
    assert s.scheduler_interval_seconds == 1.0
    assert s.scheduler_batch_size == 100
    assert s.max_backoff_seconds == 300.0


def test_settings_env_prefix():
    """Verify settings use JQ_ env prefix."""
    assert Settings.model_config["env_prefix"] == "JQ_"


def test_create_app():
    """Verify FastAPI app is created with correct metadata."""
    test_app = create_app()
    assert test_app.title == "Distributed Job Queue"
    assert test_app.version == "0.1.0"


def test_app_instance_exists():
    """Verify the module-level app instance is a FastAPI app."""
    assert app is not None
    assert app.title == "Distributed Job Queue"


def test_database_module_imports():
    """Verify database module components are importable."""
    from src.database import async_session_factory, close_db, engine, init_db

    assert engine is not None
    assert async_session_factory is not None
    assert callable(init_db)
    assert callable(close_db)


def test_redis_module_imports():
    """Verify redis client module components are importable."""
    from src.redis_client import (
        check_redis_health,
        close_redis,
        get_redis_client,
        get_redis_pool,
        init_redis,
    )

    assert callable(get_redis_pool)
    assert callable(get_redis_client)
    assert callable(init_redis)
    assert callable(close_redis)
    assert callable(check_redis_health)
