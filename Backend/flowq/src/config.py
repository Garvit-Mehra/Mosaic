"""Application configuration using pydantic-settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    app_name: str = "Distributed Job Queue"
    debug: bool = False

    # PostgreSQL
    postgres_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/jobqueue"

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    redis_max_connections: int = 20
    redis_socket_timeout: float = 3.0
    redis_socket_connect_timeout: float = 3.0

    # Worker configuration
    worker_count: int = 4
    heartbeat_interval_seconds: int = 5
    heartbeat_ttl_seconds: int = 15
    failure_check_interval_seconds: int = 5

    # Job configuration
    payload_size_limit_bytes: int = 1_048_576  # 1MB default
    default_job_timeout_seconds: int = 300
    max_job_timeout_seconds: int = 86400
    default_max_retries: int = 3
    max_retry_limit: int = 100
    max_priority: int = 10000
    default_retry_backoff_base: float = 2.0
    max_backoff_seconds: float = 300.0

    # Scheduler
    scheduler_interval_seconds: float = 1.0
    scheduler_batch_size: int = 100

    # PostgreSQL failure handling
    postgres_connect_timeout: float = 5.0

    model_config = SettingsConfigDict(
        env_prefix="JQ_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


# Global settings instance
settings = Settings()
