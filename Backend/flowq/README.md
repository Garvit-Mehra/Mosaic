# Distributed Job Queue & Task Scheduler

A fault-tolerant, horizontally-scalable distributed job queue and task scheduler built in Python. This system processes background jobs asynchronously through a priority-based queue with configurable retry policies, worker health monitoring, and automatic failure recovery.

---

## Why This Architecture?

The system solves the problem of reliable asynchronous task execution at scale. Key design decisions:

1. **Dual-storage (Redis + PostgreSQL)** — Redis provides fast O(log N) queue operations via sorted sets, while PostgreSQL ensures durability. PostgreSQL is always the source of truth; Redis is reconstructable from it.
2. **Coordinator-Worker pattern** — Separates job lifecycle management (coordinator) from execution (workers), enabling independent scaling.
3. **Heartbeat-based failure detection** — Workers signal liveness every 5 seconds. If 3 heartbeats are missed (15s TTL), the worker is declared dead and its jobs are recovered.
4. **Distributed locking** — Redis SET NX with TTL prevents two workers from executing the same job simultaneously.

---

## System Architecture

```
Client Applications
       │
       ▼ HTTP
┌───────────────────┐
│  FastAPI REST API │
└────────┬──────────┘
         │
         ▼
┌──────────────────┐       ┌───────────────┐
│  Job Coordinator │──────▶│  PostgreSQL   │
│  (state machine) │       │  (truth store)│
└────────┬─────────┘       └───────────────┘
         │
         ▼
┌──────────────────┐
│   Redis          │
│  • Priority Queue│  (sorted set: score = -priority × 1e12 + timestamp_ms)
│  • Schedule Set  │  (sorted set: score = execute_at_ms)
│  • Retry Queue   │  (sorted set: score = next_retry_ms)
│  • Dead-Letter Q │  (sorted set: score = failed_at_ms)
│  • Heartbeats    │  (string keys with 15s TTL)
│  • Locks         │  (SET NX with TTL)
└────────┬─────────┘
         │ BZPOPMIN (blocking dequeue)
         ▼
┌─────────────────┐
│   Worker Pool   │  (multiprocessing — true parallelism)
│  ┌───┐ ┌───┐    │
│  │W1 │ │W2 │... │  Each worker: own asyncio event loop
│  └───┘ └───┘    │
└─────────────────┘
         │
    Heartbeats + Results
         │
         ▼
┌──────────────────┐
│ Failure Detector │  (checks every 5s for expired heartbeats)
│ Scheduler Loop   │  (promotes due jobs every 1s)
└──────────────────┘
```

---

## Job Lifecycle (State Machine)

```
PENDING ──┬──▶ QUEUED ──▶ RUNNING ──┬──▶ COMPLETED
          │                          │
          ├──▶ SCHEDULED ──▶ QUEUED  ├──▶ FAILED ──┬──▶ QUEUED (retry)
          │                          │              │
          └──▶ CANCELLED             │              └──▶ DEAD_LETTER
                                     │
               SCHEDULED ──▶ CANCELLED
               QUEUED ──▶ CANCELLED
```

**Terminal states** (no outgoing transitions): COMPLETED, CANCELLED, DEAD_LETTER

**Key invariant**: Every transition is validated by the state machine before it's applied. Invalid transitions are rejected without modifying the job.

---

## Core Algorithms

### Priority Queue Scoring

```python
score = -priority × 1,000,000,000,000 + enqueued_at_timestamp_ms
```

- Higher priority → lower score → popped first by Redis ZPOPMIN
- Same priority → earlier enqueue time wins (FIFO)
- The 1e12 multiplier ensures priority always dominates over timestamp

### Exponential Backoff

```python
delay = min(base^retry_count, max_delay) + jitter
jitter = random(0, 0.1 × base^retry_count)
```

- Prevents thundering herd on transient failures
- Monotonically non-decreasing until cap (300s default)
- Each job configures its own `retry_backoff_base` and `max_retries`

### Distributed Lock (Redis SET NX)

```
SET lock:job:{job_id} {worker_id} NX EX {timeout + 30}
```

- Mutual exclusion: only one worker executes a job at a time
- Auto-expires on crash (prevents deadlock)
- Released via Lua script (atomic compare-and-delete) — only the holder can release

### Failure Detection

```
Every 5 seconds:
  For each ACTIVE worker in PostgreSQL:
    If heartbeat:{worker_id} key is missing in Redis:
      → Worker is dead
      → Recover all its RUNNING jobs (re-queue or DLQ)
```

- Detection time: ≤ 20 seconds (15s TTL + 5s check interval)
- Race-safe: only recovers jobs still in RUNNING state (prevents double-recovery)

---

## API Endpoints

| Method | Path | Description | Status Codes |
|--------|------|-------------|-------------|
| POST | /jobs | Submit a new job | 201, 400, 413 |
| GET | /jobs/:id | Get job status/metadata | 200, 404 |
| GET | /jobs | List jobs (filter by status) | 200, 400 |
| GET | /jobs/:id/history | Execution attempt history | 200, 404 |
| DELETE | /jobs/:id | Cancel a job | 200, 404, 409 |
| GET | /dlq | List dead-lettered jobs | 200 |
| POST | /dlq/:id/retry | Retry a DLQ job | 200, 404, 409 |
| GET | /workers | Worker status list | 200 |
| GET | /metrics | Queue depth, throughput, latency | 200 |

---

## Error Resilience Strategy

| Failure | Behavior |
|---------|----------|
| Redis down | API returns 503 for writes (submit, cancel). Reads still work from PostgreSQL. Workers pause with exponential backoff reconnection. |
| PostgreSQL down | API returns 503 for everything. Workers finish current job, hold result in memory until PG recovers. |
| Worker crash | Heartbeat expires after 15s. Failure detector re-queues abandoned jobs within 20s. |
| Redis data loss | Full state reconstructed from PostgreSQL on startup (priority queue, schedule set, retry queue). |
| Double-dequeue race | Distributed lock ensures only one worker proceeds; the other skips and dequeues next. |

---

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| API | FastAPI | Async-native, automatic OpenAPI docs, Pydantic validation |
| Queue | Redis sorted sets | O(log N) insert/pop, atomic operations, BZPOPMIN for blocking |
| Storage | PostgreSQL + SQLAlchemy async | ACID durability, complex queries, asyncpg for performance |
| Workers | Python multiprocessing | True parallelism (bypasses GIL), each with asyncio loop |
| Config | pydantic-settings | Type-safe env vars, .env file support, validation |
| Testing | pytest + Hypothesis | Property-based testing for correctness proofs |

---

## Code Walkthrough (Implemented So Far)

### `src/config.py` — Centralized Configuration

```python
class Settings(BaseSettings):
    model_config = {"env_prefix": "JQ_", "env_file": ".env"}
```

All settings are loaded from environment variables with the `JQ_` prefix (e.g., `JQ_REDIS_URL`). This enables different configs for dev/staging/prod without code changes. Key settings:

- `redis_socket_timeout: 3.0` — matches Requirement 15.1 (3-second timeout for Redis health checks)
- `postgres_connect_timeout: 5.0` — matches Requirement 15.3 (5-second timeout for PG)
- `payload_size_limit_bytes: 1MB` — enforced at submission time
- `heartbeat_interval_seconds: 5` / `heartbeat_ttl_seconds: 15` — 3 missed beats = dead

### `src/main.py` — Application Factory Pattern

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()     # verify PG connectivity
    await init_redis()  # verify Redis connectivity
    yield
    await close_redis()
    await close_db()
```

Uses FastAPI's lifespan context manager (replaces deprecated `on_event`). The app won't start serving requests until both datastores are verified healthy. On shutdown, connections are drained cleanly.

### `src/database.py` — Async SQLAlchemy Engine

```python
engine = create_async_engine(
    settings.postgres_url,
    pool_size=10,          # base connections in pool
    max_overflow=20,       # extra connections under load
    pool_pre_ping=True,    # detect stale connections
    pool_recycle=3600,     # recycle connections hourly
)
```

- **pool_pre_ping**: Before using a connection from the pool, sends a lightweight query to verify it's alive. Prevents "connection reset" errors.
- **expire_on_commit=False**: Objects remain usable after commit without re-querying. Important for returning job data in API responses.
- **get_session()**: An async generator for use as a FastAPI dependency.

### `src/redis_client.py` — Redis Connection Management

```python
_redis_pool = redis.ConnectionPool.from_url(
    settings.redis_url,
    max_connections=20,
    decode_responses=True,  # returns strings instead of bytes
)
```

- **Singleton pattern**: Pool and client are module-level singletons, initialized lazily.
- **check_redis_health()**: Used by API endpoints to return 503 when Redis is unreachable (Requirement 15.1).
- **decode_responses=True**: All Redis values come back as Python strings — important since job IDs are UUID strings.

---

## How It All Fits Together (Request Flow)

```
1. Client POSTs /jobs with {job_type: "send_email", payload: {...}, priority: 5}
2. API validates input (type registered? priority in range? payload < 1MB?)
3. Job Coordinator creates row in PostgreSQL (status = PENDING)
4. If no execute_at: Coordinator adds job_id to Redis sorted set with priority score, sets status = QUEUED
5. Worker's BZPOPMIN returns the job_id (blocks up to 5s if queue empty)
6. Worker acquires distributed lock (SET NX) → if fails, skip to next job
7. Worker sets status = RUNNING, records started_at
8. Worker executes handler with asyncio.wait_for(timeout)
9. On success: status = COMPLETED, result stored, lock released
10. On failure: retry count checked → re-queue with backoff OR move to DLQ
11. Throughout: worker sends heartbeat every 5s
12. If worker crashes: heartbeat expires → Failure Detector recovers job
```

---

## Interview Talking Points

### "Why Redis + PostgreSQL instead of just one?"
Redis gives sub-millisecond queue operations and atomic primitives (BZPOPMIN, SET NX). PostgreSQL gives ACID durability and complex queries. Redis is ephemeral/reconstructable — if it crashes, we rebuild from PG. This separation lets each tool do what it's best at.

### "How do you prevent a job from running twice?"
Distributed lock via Redis SET NX. Lock key = `lock:job:{job_id}`, value = worker_id, TTL = timeout + 30s. Released via Lua script that atomically checks the value matches before deleting. If the lock expires while the worker is still running, the worker detects it can't release and discards its result.

### "What happens when a worker crashes mid-job?"
The worker's heartbeat key in Redis has a 15-second TTL. If 3 heartbeats are missed, the key expires. The Failure Detector (running every 5s) notices the missing key, finds all RUNNING jobs assigned to that worker, and either re-queues them (if retries remain) or moves them to DLQ.

### "How do you handle the thundering herd problem on retries?"
Exponential backoff with jitter. The jitter (10% of the delay) spreads retry attempts across time so multiple failed jobs don't all retry simultaneously and overwhelm the system.

### "Why multiprocessing instead of asyncio-only workers?"
Python's GIL prevents true CPU parallelism with threads. Multiprocessing gives each worker its own Python interpreter. Each worker then runs an asyncio event loop for I/O-bound operations (network calls, DB queries). This gives both CPU parallelism and I/O concurrency.

### "How is the queue state durable if Redis is in-memory?"
PostgreSQL is the source of truth for all job state. On startup (or Redis recovery), we reconstruct Redis data structures from PG: all QUEUED jobs go into the priority sorted set, all SCHEDULED jobs go into the schedule set, and RUNNING jobs are treated as abandoned and recovered. This is the "state reconstruction" capability.

---

## Project Structure

```
src/
├── __init__.py
├── config.py          # Settings class (pydantic-settings, env vars)
├── main.py            # FastAPI app factory + lifespan handler
├── database.py        # Async SQLAlchemy engine + session factory
├── redis_client.py    # Async Redis connection pool + health check
├── api/               # REST API route handlers
├── core/              # Business logic (coordinator, state machine, locks, backoff)
├── models/            # SQLAlchemy ORM models
├── workers/           # Worker process + pool management
├── scheduler/         # Scheduled/retry job promotion loop
└── metrics/           # Throughput, latency, queue depth tracking
tests/
└── ...                # pytest + hypothesis property tests
```

---

## Running the Project

```bash
# Install dependencies
pip install -e ".[dev]"

# Set environment variables (or use .env file)
export JQ_POSTGRES_URL="postgresql+asyncpg://postgres:postgres@localhost:5432/jobqueue"
export JQ_REDIS_URL="redis://localhost:6379/0"

# Run the API server
uvicorn src.main:app --host 0.0.0.0 --port 8000

# Run tests
pytest
```

---

## Implementation Progress

- [x] Project scaffold, dependencies, configuration
- [ ] SQLAlchemy models + Alembic migrations
- [ ] Job state machine with transition validation
- [ ] Priority queue (Redis sorted set operations)
- [ ] Distributed lock (SET NX + Lua release)
- [ ] Job submission, validation, routing
- [ ] Retry logic + exponential backoff
- [ ] Dead-letter queue management
- [ ] Worker process + heartbeat system
- [ ] Scheduler promotion loop
- [ ] Failure detection + recovery
- [ ] REST API endpoints
- [ ] Metrics collection
- [ ] Error resilience (reconnection, state reconstruction)
