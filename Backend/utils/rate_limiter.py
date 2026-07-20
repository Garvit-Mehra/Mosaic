"""
Mosaic Rate Limiter

Supports two backends:
- Redis (production): persistent, shared across workers/processes
- In-memory (development): no dependencies needed

Controlled by REDIS_URL environment variable:
- If set: uses Redis
- If not set: falls back to in-memory (lost on restart)
"""

import os
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict

from utils.logger import get_logger

logger = get_logger("rate_limiter")


class RateLimiter(ABC):
    """Abstract rate limiter interface."""

    @abstractmethod
    def check(self, key: str) -> bool:
        """Returns True if request is allowed, False if rate limited."""
        pass

    @abstractmethod
    def remaining(self, key: str) -> int:
        """Returns remaining attempts for this key."""
        pass

    @abstractmethod
    def reset_time(self, key: str) -> int:
        """Returns seconds until the rate limit window resets."""
        pass


class MemoryRateLimiter(RateLimiter):
    """In-memory rate limiter. Good for dev, lost on restart."""

    def __init__(self, max_attempts: int, window_seconds: int):
        self.max_attempts = max_attempts
        self.window = window_seconds
        self._attempts: Dict[str, list] = defaultdict(list)

    def _clean(self, key: str):
        now = time.time()
        self._attempts[key] = [t for t in self._attempts[key] if now - t < self.window]

    def check(self, key: str) -> bool:
        self._clean(key)
        if len(self._attempts[key]) >= self.max_attempts:
            return False
        self._attempts[key].append(time.time())
        return True

    def remaining(self, key: str) -> int:
        self._clean(key)
        return max(0, self.max_attempts - len(self._attempts[key]))

    def reset_time(self, key: str) -> int:
        self._clean(key)
        if not self._attempts[key]:
            return 0
        oldest = min(self._attempts[key])
        return max(0, int(self.window - (time.time() - oldest)))


class RedisRateLimiter(RateLimiter):
    """Redis-backed rate limiter. Persistent, shared across workers."""

    def __init__(self, max_attempts: int, window_seconds: int, redis_url: str):
        import redis
        self.max_attempts = max_attempts
        self.window = window_seconds
        self.redis = redis.from_url(redis_url, decode_responses=True)
        self._prefix = "mosaic:ratelimit:"
        logger.info(f"Redis rate limiter connected: {redis_url}")

    def _key(self, key: str) -> str:
        return f"{self._prefix}{key}"

    def check(self, key: str) -> bool:
        rkey = self._key(key)
        now = time.time()

        pipe = self.redis.pipeline()
        # Remove old entries
        pipe.zremrangebyscore(rkey, 0, now - self.window)
        # Count current entries
        pipe.zcard(rkey)
        # Add current attempt
        pipe.zadd(rkey, {str(now): now})
        # Set expiry on the key
        pipe.expire(rkey, self.window)
        results = pipe.execute()

        count = results[1]  # zcard result before adding
        if count >= self.max_attempts:
            # Remove the entry we just added
            self.redis.zrem(rkey, str(now))
            return False
        return True

    def remaining(self, key: str) -> int:
        rkey = self._key(key)
        now = time.time()
        self.redis.zremrangebyscore(rkey, 0, now - self.window)
        count = self.redis.zcard(rkey)
        return max(0, self.max_attempts - count)

    def reset_time(self, key: str) -> int:
        rkey = self._key(key)
        entries = self.redis.zrange(rkey, 0, 0, withscores=True)
        if not entries:
            return 0
        oldest_score = entries[0][1]
        return max(0, int(self.window - (time.time() - oldest_score)))


def create_rate_limiter(max_attempts: int = 5, window_seconds: int = 300) -> RateLimiter:
    """
    Factory function — creates the appropriate rate limiter based on environment.
    Uses Redis if REDIS_URL is set, otherwise falls back to in-memory.
    """
    redis_url = os.getenv("REDIS_URL")

    if redis_url:
        try:
            limiter = RedisRateLimiter(max_attempts, window_seconds, redis_url)
            return limiter
        except Exception as e:
            logger.warning(f"Failed to connect to Redis ({e}). Falling back to in-memory rate limiter.")

    logger.info("Using in-memory rate limiter (set REDIS_URL for Redis)")
    return MemoryRateLimiter(max_attempts, window_seconds)
