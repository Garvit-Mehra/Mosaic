<div align="center">

# Mosaic — System Design

</div>

---

## System Overview

Mosaic is a multi-agent AI assistant with a distributed task processing backend. The system has two major subsystems:

1. **Chat System** — real-time agent routing, streaming LLM responses, conversation persistence
2. **FlowQ** — fault-tolerant distributed job processing for long-running operations

Both share PostgreSQL and Redis infrastructure.

```mermaid
graph TB
    subgraph "Client Layer"
        Browser[Browser]
    end

    subgraph "Frontend (Next.js)"
        NextAuth[NextAuth v5]
        Pages[Chat / Settings / Admin]
        Middleware[Route Protection]
    end

    subgraph "Backend (FastAPI)"
        API[REST API]
        AgentRouter[Agent Router / Classifier]
        Handler[Stateless Chat Handler]
        JobBridge[FlowQ Bridge]
    end

    subgraph "AI Layer"
        LLM[LLM Provider]
        MCP[MCP Tool Servers]
        RAG[RAG / Vector Store]
    end

    subgraph "FlowQ"
        Coordinator[Job Coordinator]
        Scheduler[Scheduler Loop]
        Workers[Worker Pool]
        FailureDetector[Failure Detector]
    end

    subgraph "Data Layer"
        PG[(PostgreSQL)]
        Redis[(Redis)]
    end

    Browser --> NextAuth
    NextAuth --> Pages
    Pages --> API
    Middleware --> Pages

    API --> BGCheck[Background Task Detector]
    BGCheck -->|"background op"| JobBridge
    BGCheck -->|"normal message"| AgentRouter
    AgentRouter --> Handler
    Handler --> LLM
    Handler --> MCP
    Handler --> RAG

    JobBridge --> Coordinator
    Coordinator --> Redis
    Coordinator --> PG
    Scheduler --> Redis
    Scheduler --> PG
    Workers --> Redis
    Workers --> PG
    Workers --> RAG
    Workers --> MCP
    FailureDetector --> Redis
    FailureDetector --> PG

    Handler --> PG
    API --> Redis
```

---

## Chat System

### Request Flow

```mermaid
sequenceDiagram
    participant C as Client
    participant F as Frontend
    participant B as Backend
    participant BG as Background Check
    participant CL as Classifier LLM
    participant A as Agent
    participant TQ as FlowQ
    participant DB as PostgreSQL

    C->>F: Send message
    F->>B: POST /chat/stream (Bearer token)
    B->>DB: Load conversation history
    B->>BG: Check for background patterns

    alt Background task detected (file load, URL scrape)
        BG->>TQ: Submit job (rag_process / web_scrape)
        TQ->>DB: Create job record (QUEUED)
        TQ-->>BG: job_id
        BG-->>B: Immediate response with job_id
        B-->>F: "Processing in background (job: abc-123)"
        F-->>C: Display status message

        Note over TQ: Worker picks up job asynchronously
        C->>F: "check job abc-123"
        F->>B: POST /chat/stream
        B->>BG: Detect job status query
        BG->>TQ: Get job result
        TQ-->>BG: {status: completed, result: {...}}
        BG-->>B: Format result
        B-->>F: SSE response
        F-->>C: Display result

    else Normal chat message
        BG-->>B: None (no background task)
        B->>CL: Classify query
        CL-->>B: agent name
        B->>A: Invoke agent with context
        A-->>B: Stream tokens
        B-->>F: SSE (token events)
        F-->>C: Render tokens in real-time
        B->>DB: Persist messages
    end
```

### Agent Registry

Agents are initialized once at startup and shared across requests (stateless):

| Agent | Tools | Purpose |
|-------|-------|---------|
| general | None | Writing, coding, math, creative tasks |
| web | TavilySearch | Live info (news, weather, scores) |
| rag | load_document, query_documents | Document Q&A |
| {mcp_server} | Dynamic from server | Custom tool execution |

### Classification

The classifier LLM receives:
- Available agents and their descriptions
- The user's current message
- Rules: "when in doubt, use general"

Returns a single agent name. The chosen agent then processes the full message with conversation history as context.

### Conversation Persistence

- Messages stored in PostgreSQL (`conversations` + `messages` tables)
- Each request loads the last N messages from DB (no in-memory state)
- Ownership enforced: users only see their own conversations
- Auto-creates conversation on first message

---

## FlowQ

### Purpose

Offloads long-running operations from the synchronous chat flow:
- MCP tool calls that take > 5 seconds
- PDF processing and RAG document indexing
- Web scraping
- Batch message processing

The chat handler automatically detects these operations via pattern matching on the user message. When detected, the job is submitted to the queue and the user receives an immediate acknowledgment with a job ID. Normal conversational messages bypass the queue entirely.

**Auto-detected patterns:**

| User message pattern | Job type submitted |
|---------------------|-------------------|
| "load document 'file.pdf'" | `rag_process` |
| "process pdf 'report.pdf'" | `rag_process` |
| "scrape https://example.com" | `web_scrape` |
| "fetch url https://..." | `web_scrape` |
| "check job abc-123-..." | Queries job status (no submission) |
| Everything else | No queue — processed by LLM directly |

### Job Lifecycle

```mermaid
stateDiagram-v2
    [*] --> PENDING: submit
    PENDING --> QUEUED: immediate
    PENDING --> SCHEDULED: has execute_at
    PENDING --> CANCELLED: cancel

    SCHEDULED --> QUEUED: scheduler promotes
    SCHEDULED --> CANCELLED: cancel

    QUEUED --> RUNNING: worker dequeues
    QUEUED --> CANCELLED: cancel

    RUNNING --> COMPLETED: success
    RUNNING --> FAILED: error / timeout

    FAILED --> QUEUED: retry (backoff)
    FAILED --> DEAD_LETTER: max retries exceeded

    DEAD_LETTER --> QUEUED: manual retry

    COMPLETED --> [*]
    CANCELLED --> [*]
```

### Priority Queue

Redis sorted set (`queue:priority`) with score:

```
score = -priority * 1,000,000,000,000 + enqueued_at_ms
```

- Higher priority produces lower score, popped first by `BZPOPMIN`
- Equal priority = FIFO by enqueue time
- Additional sorted sets: `queue:scheduled`, `queue:retry`, `queue:dlq`

### Worker Architecture

```mermaid
graph LR
    subgraph "Worker Process"
        Poll[Poll Loop]
        Heartbeat[Heartbeat Task]
        FlushTask[PG Flush Task]
    end

    Poll -->|BZPOPMIN| Redis[(Redis)]
    Poll -->|acquire lock| Redis
    Poll -->|execute| Handler[Job Handler]
    Poll -->|persist result| PG[(PostgreSQL)]
    Heartbeat -->|SET EX 15s| Redis
    FlushTask -->|retry pending| PG
```

Each worker:
1. Registers in PostgreSQL (ACTIVE status, hostname, PID)
2. Sends heartbeat every 5s (Redis key with 15s TTL)
3. Blocking dequeue from priority queue (5s timeout)
4. Acquires distributed lock (Redis SET NX, TTL = job_timeout + 30s)
5. Executes handler with `asyncio.wait_for(timeout)`
6. Persists result to PostgreSQL
7. Releases lock in `finally` block

### Failure Detection

```mermaid
graph TD
    FD[Failure Detector Loop<br/>every 5s] -->|query ACTIVE workers| PG[(PostgreSQL)]
    FD -->|check heartbeat key| Redis[(Redis)]
    FD -->|missing heartbeat| Dead{Worker Dead}
    Dead -->|retry_count < max| Requeue[Re-queue Job]
    Dead -->|retries exhausted| DLQ[Dead Letter Queue]
    Dead -->|mark worker| DeadStatus[Worker = DEAD]
```

- Detection time: < 20 seconds (15s heartbeat TTL + 5s check interval)
- Uses `SELECT ... FOR UPDATE` to prevent double-recovery
- Abandoned RUNNING jobs are re-queued with incremented retry_count

### Retry with Exponential Backoff

```
delay = min(base^retry_count, 300s) + random(0, 0.1 * delay)
```

- Default base: 2.0 (2s, 4s, 8s, 16s, 32s, 64s, 128s, 256s, 300s cap)
- Jitter prevents thundering herd
- Failed jobs go to `queue:retry` sorted set with score = next_retry_at_ms
- Scheduler promotes when backoff elapses

### State Reconstruction

On startup or Redis recovery:
1. Clear all Redis queue keys
2. Rebuild `queue:priority` from all QUEUED jobs in PostgreSQL
3. Rebuild `queue:scheduled` from SCHEDULED jobs
4. Recover RUNNING jobs (treated as abandoned: re-queue or DLQ)
5. Block new submissions until complete

Ensures the system recovers fully from any restart or Redis failure.

### Distributed Lock

```
Key:    lock:job:{job_id}
Value:  worker_id
TTL:    job_timeout + 30s
```

- `SET NX` for acquisition (atomic, no race conditions)
- Lua script for release (compare-and-delete, only holder can release)
- Prevents double-execution when recovery re-queues a job still being processed

---

## Data Layer

### PostgreSQL Tables

| Table | Purpose |
|-------|---------|
| `conversations` | Chat conversation metadata (title, user_id, timestamps) |
| `messages` | Individual messages (role, content, agent, timestamp) |
| `users` | User accounts (username, email, password_hash, role, verified) |
| `user_mcp_servers` | Per-user MCP server configurations |
| `jobs` | Job state, payload, result, retry config |
| `job_executions` | Execution history per attempt |
| `workers` | Worker registration and status |

### Redis Keys

| Key Pattern | Type | Purpose |
|-------------|------|---------|
| `queue:priority` | Sorted Set | Active job queue |
| `queue:scheduled` | Sorted Set | Future-scheduled jobs |
| `queue:retry` | Sorted Set | Jobs waiting for backoff |
| `queue:dlq` | Sorted Set | Dead-letter queue |
| `heartbeat:{worker_id}` | String (TTL) | Worker liveness |
| `lock:job:{job_id}` | String (TTL) | Distributed execution lock |
| `metrics:latency_samples` | Sorted Set | 60s sliding window of latency data |
| `metrics:completed_count` | String | Total completed counter |
| `mosaic:ratelimit:*` | Sorted Set | Rate limiter windows |

---

## Authentication

```mermaid
graph LR
    User -->|credentials or OAuth| NextAuth
    NextAuth -->|session cookie| Browser
    NextAuth -->|POST /auth/oauth| Backend
    Backend -->|JWT access_token| NextAuth
    Browser -->|Bearer token| Backend
    Backend -->|verify JWT| Response
```

- Sessions: httpOnly signed cookies (XSS-safe)
- Backend tokens: JWT with type, expiry, role
- Rate limiting: 5 login attempts / 5 min per IP
- SSRF protection: blocks private IPs on MCP server URLs (temporarily disabled for local dev)
- Passwords: bcrypt with salt

---

## Monitoring & Metrics

Collected in real-time via Redis sorted sets (60s sliding window):

| Metric | Source |
|--------|--------|
| Queue depth | `ZCARD queue:priority` |
| Active workers | PostgreSQL count (status=ACTIVE) |
| Jobs/second | Completed count in last 60s / 60 |
| Latency P50/P95 | Sorted latency samples, nearest-rank percentile |
| DLQ size | `ZCARD queue:dlq` |

Exposed via `GET /flowq/metrics` and visible in the admin panel.

---

## Scalability Properties

| Property | How |
|----------|-----|
| Horizontal worker scaling | Add workers without architecture changes |
| Stateless API | No in-memory conversation state, DB loaded per request |
| Connection pooling | SQLAlchemy QueuePool (10 + 20 overflow) |
| Multi-instance scheduler | Atomic ZREM prevents double-promotion |
| Multi-instance failure detector | SELECT FOR UPDATE prevents double-recovery |
| Redis as coordination layer | All queue ops are atomic (ZADD, BZPOPMIN, ZREM) |
| PostgreSQL as source of truth | Redis is fully reconstructable from PG |
