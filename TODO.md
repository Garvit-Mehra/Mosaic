# Mosaic — TODO

---

## 1. Security

### Completed
- [x] Input length validation — reject messages over 10,000 chars
- [x] Streaming timeout — kill SSE connection after 120s
- [x] Rate limit on /chat — 20 messages per user per minute
- [x] Conversation ownership check on POST /chat
- [x] Sanitize MCP server URLs — only http/https
- [x] Graceful error responses — no tracebacks to client
- [x] Frontend error boundary
- [x] Request body size limit — 1MB max
- [x] SSRF protection — blocks private/internal IPs
- [x] Rate limit on /auth/register — 3 per hour per IP
- [x] Secure /auth/oauth — shared secret header

### Remaining
- [ ] Account lockout — lock after N total failed login attempts
- [x] Password strength check — require mixed chars, not just length >= 6
- [x] Security headers middleware — X-Content-Type-Options, X-Frame-Options, HSTS
- [ ] Audit log for admin actions
- [ ] Session revocation — token blacklist via Redis
- [ ] Input sanitization — strip XSS from titles, descriptions, server names
- [ ] Conversation storage limit — cap max messages per user
- [ ] CORS validation in production — fail at startup if ALLOWED_ORIGINS is "*" and ENVIRONMENT=production

---

## 2. Authentication (Critical Fixes)

### Token Lifecycle
- [ ] Backend token auto-refresh — NextAuth's jwt() callback must check token expiry and call /auth/refresh before it expires (currently token dies after 24h but session lives 7 days)
- [ ] Single-use refresh tokens — rotate on each refresh call to prevent replay
- [ ] Add aud/iss claims to JWT — prevent cross-instance token reuse

### Access Control
- [x] Enforce verified=True for login — unverified users can currently log in
- [x] JWT role invalidation — demoting a user doesn't revoke existing tokens
- [x] /auth/oauth wide open when OAUTH_SHARED_SECRET is empty — must fail closed (reject all if no secret configured)
- [ ] Password change invalidates existing tokens
- [x] Remove EnvUserProvider fallback in production — env-based login should be disabled when DB has users

### Missing Flows
- [ ] Password reset flow — "forgot password" does not exist
- [ ] Email OTP verification — replace placeholder with actual email sending
- [ ] Email uniqueness real-time check on register form
- [ ] Token rotation on role change

### Hardening
- [ ] Login response should not return refresh token in JSON body — use httpOnly cookie instead
- [ ] No login event logging (geolocation/device)
- [ ] No "active sessions" view or revocation
- [ ] No CAPTCHA on register — rate limit alone may not stop bots

---

## 3. Functionality

### Backend Bugs
- [x] Fix auto-rename — creates orphan conversations (sends POST /chat with null conv_id)
- [x] Singleton UserManager — re-instantiated per request, leaks connection pools
- [x] Agent registry per-user isolation — user A's servers route queries for user B
- [x] Detect client disconnect on streaming — generator keeps running after user navigates away

### Features
- [ ] Pagination on GET /conversations — offset/cursor for large histories
- [ ] Message limit per conversation — truncate oldest on overflow
- [ ] Authenticated remote MCP servers — add per-user OAuth 2.0 authorization, encrypted token storage/refresh, authenticated MCP requests, and a "sign in required" connection state (for example, Google Calendar MCP)
- [ ] Admin panel "verify user" button — manage_users.py is the only way currently
- [x] Cascade delete — deleting user should delete their conversations, messages, MCP configs
- [ ] Job progress reporting — 0-100% updates from FlowQ workers

### FlowQ Integration
- [ ] Register handlers with worker's HandlerRegistry — workers can't execute jobs without this
- [ ] Start FlowQ workers in start.sh — jobs queue up but nothing processes them
- [ ] Run Alembic migrations on startup or via setup script — FlowQ tables don't exist without manual migration
- [ ] Proper package imports — replace sys.path hack with pip install -e or PYTHONPATH
- [x] RAG vector store persistence — migrated to ChromaDB for persistence
- [ ] Wire handlers to live MCP/RAG — end-to-end test with real servers
- [ ] Pre-warm batch_chat agent registry in workers
- [ ] Dead-letter queue visibility in admin panel
- [ ] FlowQ and Mosaic share same PG but different Base metadata — ensure both table sets are created

---

## 4. Error Handling & User Feedback

### Frontend
- [x] Parse backend error responses — read `detail` from JSON, show actual message
- [ ] Detect incomplete streams — SSE ends with empty content, show retry
- [ ] Explicit fetch timeout (AbortController, 30s) — show "Request timed out"
- [x] Handle 429 responses — show "Too many messages, slow down" not generic error
- [ ] Session expired notification — message before redirect to login
- [ ] Admin panel error states — "Failed to load" with retry button
- [x] Network error distinction — "backend not running" vs "network issue" vs "server error"

### Backend
- [ ] Specific error for LLM unreachable — "AI model not available. Ensure Ollama is running."
- [ ] Specific error for DB unreachable — "Service temporarily unavailable."
- [ ] Clean MCP tool call failures — catch LangChain tracebacks
- [ ] FlowQ unavailable — clean 503 instead of import error traceback

---

## 5. Quality of Life (UI/UX)

### Completed
- [x] Markdown rendering in chat
- [x] Copy message button
- [x] Code syntax highlighting
- [x] Auto-scroll toggle
- [x] Conversation auto-rename
- [x] Retry failed messages
- [x] Dark/light theme toggle
- [x] Search conversations in sidebar
- [x] Typing indicator animation

### Remaining
- [ ] Token count / response time display per message
- [ ] Mobile responsive layout — sidebar as drawer on small screens
- [x] File upload for RAG — drag-and-drop PDFs into chat
- [ ] Export conversation as markdown or JSON
- [ ] Keyboard shortcuts — Cmd+K new chat, Cmd+/ settings
- [ ] User avatar in chat bubbles
- [ ] Theme flash fix — inline script sets data-theme before hydration
- [ ] Loading skeleton when switching conversations
- [ ] Sidebar refresh immediately after new conversation created

---

## 6. Production Infrastructure

### Docker & Deployment
- [ ] Fix Dockerfile COPY path — `COPY ../requirements.txt` doesn't work in Docker context
- [ ] Logging to stdout in Docker — file-based logs lost on container restart
- [ ] Container restart policy (restart: unless-stopped)
- [ ] Uvicorn worker count — currently single worker, need 2-4 for production
- [ ] HTTPS setup — SSL cert via Caddy or nginx with Let's Encrypt
- [ ] Set ENVIRONMENT=production in Docker .env

### Database & State
- [ ] Health endpoint must check DB and Redis — currently returns "ok" even if DB is dead
- [ ] Database connection error at startup — fail fast with clear message, not hang
- [ ] PostgreSQL volume persistence verification
- [ ] Redis persistence config (appendonly yes) — queue data survives restart
- [ ] Rate limiter requires Redis in production — in-memory resets on deploy
- [ ] Database backup strategy — no export/backup mechanism exists
- [ ] FlowQ job cleanup — completed/cancelled jobs accumulate forever
- [ ] Transaction isolation on conversation creation + first message

### DNS & OAuth
- [ ] Domain name and DNS setup
- [ ] SSL certificate
- [ ] Set real OAuth redirect URIs in provider dashboards
- [ ] Production .env file with real secrets, strong passwords, specific origins
- [ ] OAUTH_SHARED_SECRET must be set in both Frontend and Backend .env

### Frontend Build
- [ ] BACKEND_URL baked at build time — env var read at build, not runtime
- [ ] Frontend production build verification in Docker

### Monitoring
- [ ] Error alerting — notification when errors spike
- [ ] Disk space monitoring — databases grow unbounded
- [ ] Request rate visibility outside log files
- [ ] No graceful shutdown notification for active SSE streams

---

## 7. Testing & Documentation

### Testing
- [ ] Integration test: full auth flow (register, login, chat, logout)
- [ ] Integration test: SSE streaming end-to-end
- [ ] Load test in production-like environment
- [ ] Test OAuth flow with real providers
- [ ] Test database failover behavior

### Documentation
- [ ] Production deployment runbook — step-by-step first deploy
- [ ] Secrets rotation procedure — rotate JWT_SECRET without logging everyone out
- [ ] Incident response guide — DB full, Redis OOM, LLM unresponsive
- [ ] User-facing help — how to use MCP servers, background jobs

## System Design Patterns (useful additions)

### Reliability
- [ ] Circuit breaker on LLM calls — if Ollama fails 3 times in a row, stop calling for 30s and return "service unavailable" instead of hammering a dead server
- [ ] Circuit breaker on MCP server calls — mark server as unhealthy after N consecutive failures, auto-retry after cooldown
- [ ] Bulkhead isolation — separate thread/connection pools for chat vs admin vs FlowQ to prevent one overloaded path from killing the others
- [ ] Retry with backoff on frontend fetch — if backend returns 5xx, retry 2 times with 1s/3s delay before showing error

### Caching
- [ ] Response cache for repeated queries — cache classifier results for identical messages (same user, same context hash) to skip the classification LLM call
- [ ] Conversation context cache (Redis) — avoid re-reading last 10 messages from PG on rapid back-to-back messages in the same conversation
- [x] MCP tool list cache — cache tool listings per server (TTL 5 min) instead of fetching on every /servers/{name}/tools call

### Monitoring & Observability
- [ ] Health endpoint with dependency checks — /health returns {db: ok/down, redis: ok/down, llm: ok/down, queue_depth: N}
- [ ] Request latency histogram — track P50/P95/P99 per endpoint (not just total, but per-route)
- [ ] Active connection counter — how many SSE streams are currently open

### Data Management
- [ ] Event sourcing for audit log — store admin actions as immutable events (who did what, when) rather than just overwriting state
- [ ] CQRS for conversations — separate read model (fast list/search) from write model (append messages). Useful when conversation list queries slow down at scale
- [ ] TTL-based job archival — move completed/cancelled FlowQ jobs to an archive table after 7 days to keep the active table small

### Async & Messaging
- [ ] Pub/Sub for real-time notifications — when a background job completes, publish to a Redis channel that the frontend subscribes to (instead of polling)
- [ ] Webhook callbacks for job completion — allow users to register a URL that gets called when their job finishes
- [ ] Back-pressure signal to frontend — when queue depth > threshold, respond with estimated wait time instead of immediate processing

### Security Patterns
- [ ] Valet key for file uploads — generate short-lived presigned URLs for direct-to-storage uploads instead of streaming through the backend
- [ ] Gateway rate limiting per tier — different rate limits for free vs premium users (admin gets unlimited)

## Microservices Migration Roadmap

> Each phase depends on the previous. Do not skip phases.

### Phase 0: Prerequisites (must complete before any split)
- [ ] All Production Blockers (section 6) resolved
- [ ] All Auth Critical Fixes (section 2) resolved
- [ ] FlowQ handlers registered and workers functional
- [ ] Integration tests passing for auth, chat, and jobs
- [ ] Docker Compose working end-to-end with health checks
- [ ] Shared JWT_SECRET validation documented (any service can verify tokens independently)
- [ ] Internal API contracts defined (what each future service exposes)

### Phase 1: Extract FlowQ as independent service
**Prereqs**: Phase 0 complete
- [ ] Deploy FlowQ as its own Docker container (already has own FastAPI app)
- [ ] Replace `from flowq.src.mosaic_bridge import ...` with HTTP calls (`httpx.post("http://flowq:8082/jobs")`)
- [ ] FlowQ gets its own port (8082), own health endpoint
- [ ] Remove FlowQ mount from cifastapi_mosaic.py
- [ ] Update docker-compose: flowq-api service (separate from flowq-worker)
- [ ] Gateway routes /jobs/* and /flowq/* to FlowQ service
- [ ] Verify: chat still works, background jobs still process

### Phase 2: Extract Auth as independent service
**Prereqs**: Phase 1 complete
- [ ] Move `utils/auth.py`, `utils/UserDB.py` to own FastAPI app (port 8081)
- [ ] Auth service owns: users table, user_mcp_servers table
- [ ] Expose internal endpoints: `/internal/validate-token`, `/internal/get-user`
- [ ] Chat service validates tokens locally (just JWT decode with shared secret)
- [ ] Chat service calls Auth for user lookup only when needed (server list, etc.)
- [ ] Update NextAuth BACKEND_URL to point to Auth service for login/register
- [ ] Gateway routes /auth/* to Auth service
- [ ] Verify: login, register, OAuth all work through new service

### Phase 3: Service communication hardening
**Prereqs**: Phase 2 complete
- [ ] Internal API key for service-to-service calls (not exposed externally)
- [ ] Health checks between services (chat checks auth is up, etc.)
- [ ] Retry with backoff on inter-service HTTP calls
- [ ] Circuit breaker on Auth → if auth service is down, reject requests with 503 not hang
- [ ] Centralized logging — all services log to stdout, collected by Docker/ELK
- [ ] Request tracing — correlation ID passed through service chain

### Phase 4: Independent databases (optional, for true isolation)
**Prereqs**: Phase 3 stable in production for 2+ weeks
- [ ] Auth service gets its own PostgreSQL database (users, mcp_servers)
- [ ] Chat service owns conversations + messages database
- [ ] FlowQ owns jobs, executions, workers database
- [ ] Cross-service queries replaced with API calls
- [ ] Data consistency strategy: eventual consistency between services via events
- [ ] Backup strategy per database

### Phase 5: API Gateway
**Prereqs**: Phase 3 complete
- [ ] Replace nginx routing with proper API gateway (Kong, Traefik, or custom)
- [ ] Per-service rate limiting at gateway level
- [ ] Request routing rules centralized
- [ ] SSL termination at gateway
- [ ] Authentication verification at gateway (validate JWT once, pass user context to services)
- [ ] Load balancing per service (multiple instances of chat service behind gateway)
