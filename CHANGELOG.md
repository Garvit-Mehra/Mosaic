# Changelog

All notable changes to the Mosaic project will be documented in this file.


## [2.2.0] - 2025-07-20

### Added
- **PostgreSQL Support**: Production database with connection pooling (`pool_size=10`, `max_overflow=20`, `pool_pre_ping`)
- **Redis Rate Limiter**: Shared across workers, persistent across restarts (uses sorted sets)
- **LLM Provider Abstraction** (`utils/llm.py`): Single config to switch between Ollama, OpenAI, or any OpenAI-compatible API (vLLM, TGI, Groq, Together)
- **Docker Setup**: `Dockerfile` for backend and frontend, `docker-compose.yml` for full stack
- **Database Indexes**: Added composite indexes on `user_id + updated_at` and `conversation_id + timestamp`
- **MCP_SERVERS Env Var**: Server configs can now be set via JSON environment variable
- **Health Checks**: Docker services have proper health checks with dependency ordering

### Changed
- **Stateless Architecture**: Complete rewrite of `client.py` — no in-memory conversation state
  - `AgentRegistry`: Holds agents, initialized once at startup
  - `MosaicHandler`: Stateless per-request handler, loads context from DB each time
  - Safe for `uvicorn --workers N` (multiple concurrent processes)
- **ConversationDB**: Now reads `DATABASE_URL` from env — auto-switches between SQLite and PostgreSQL
- **Rate Limiter**: Moved to `utils/rate_limiter.py` with abstract interface — Redis if `REDIS_URL` set, in-memory fallback for dev
- **LLM Config**: Removed hardcoded `ChatOllama` calls — all models created via `utils/llm.py` factory
- **Requirements**: Added `psycopg2-binary`, `redis`, `langchain-text-splitters`, `langchain-core`
- **Admin Status**: Now shows LLM provider, database type, and Redis status

### Removed
- **In-Memory Conversation State**: The old `Mosaic` class with `deque` history is gone
- **`MosaicAPI` class**: Replaced by `AgentRegistry` + `MosaicHandler`
- **Per-process conversation tracking**: No more `self.conversation_id` on the API instance


## [2.1.0] - 2025-07-20

### Added
- **Full Authentication System**: NextAuth v5 with httpOnly cookie sessions
- **OAuth Providers**: Google, GitHub, and Microsoft (Azure AD) sign-in
- **Credentials Login**: Username/password validated against backend with bcrypt
- **Route Protection Middleware**: Next.js middleware blocks unauthenticated access
- **Admin Panel** (`/admin`): System diagnostics, log viewer, config inspector, danger zone
- **Admin-Only Endpoints**: `/admin/status`, `/admin/config`, `/admin/logs`, `/admin/conversations/clear`
- **Request Logs**: Separate `requests.log` with method, path, status, duration, body
- **OAuth Backend Endpoint**: `POST /auth/oauth` creates backend tokens for OAuth users
- **Token Refresh**: `POST /auth/refresh` endpoint for long-lived sessions
- **Health Check**: `GET /health` for uptime monitoring (no auth required)
- **Role-Based Access**: Admin sees all conversations; users see only their own
- **Rate Limiting on Login**: 5 attempts per IP per 5 minutes (configurable)
- **ADMIN_EMAILS Config**: Control who gets admin role via OAuth
- **SessionProvider**: Proper React context for auth state across components
- **Login Page OAuth Buttons**: Google/GitHub/Microsoft with branded SVG icons
- **Suspense Boundary**: Login page properly handles SSR with useSearchParams

### Changed
- **Auth Architecture**: Moved from localStorage JWT to httpOnly cookie sessions (XSS-safe)
- **MCP Server Access**: Now available to all authenticated users (was admin-only)
- **Sidebar**: Uses `useSession()` hook instead of localStorage reads (no hydration mismatch)
- **Settings Page**: Accessible to all users for MCP server management
- **Admin endpoints**: Moved under `/admin/` prefix (was `/logs`, `/status`)
- **CORS**: Changed `allow_credentials=False` (Bearer tokens don't need cookies from CORS)
- **Frontend .env**: Added `AUTH_SECRET`, `BACKEND_URL`, OAuth provider vars
- **Logout**: Uses `signOut()` from next-auth instead of manual localStorage clear

### Removed
- **localStorage Token Storage**: Replaced by httpOnly cookies (prevents XSS token theft)
- **Manual Auth Redirect Logic**: Handled by Next.js middleware now
- **Prisma/Database Adapter**: Removed Prisma dependency — auth is stateless JWT
- **`src/generated/prisma/`**: Deleted generated Prisma client files
- **Old `/logs` and `/status` routes**: Moved under `/admin/` prefix

### Security
- Sessions stored in signed httpOnly cookies — JavaScript cannot access them
- CSRF protection built into NextAuth
- Passwords hashed with bcrypt (salted, timing-safe)
- OAuth tokens never sent to the browser
- Rate limiting prevents brute force attacks
- Middleware blocks unauthorized route access server-side
- No secrets in source code — all in gitignored `.env` files
- `JWT_SECRET` required in production mode (fails loudly if missing)


## [2.0.0] - 2025-07-13

### Added
- **Full-Stack Application**: Next.js 15 frontend with real-time streaming chat UI
- **Streaming Responses**: Token-by-token SSE streaming from backend to frontend
- **Conversation Persistence**: SQLite-backed conversation history via SQLAlchemy
- **Settings UI**: Add/remove/monitor MCP servers from the browser (`/settings`)
- **Hot-Reload MCP Servers**: `POST /servers/refresh` picks up newly started servers without restart
- **Dynamic Server Management**: Add MCP servers at runtime via API or UI
- **Server Tools Viewer**: See all tools a connected MCP server provides
- **Conversation CRUD**: Full REST API for conversations (create, list, get, update, delete)
- **Auto-Conversation Creation**: First message auto-creates a conversation if none specified
- **Collapsible Sidebar**: Shows real conversations from the database with delete support
- **Centralized Logging System**: Rotating file logs with separate error log
- **Colored Console Output**: Level-colored terminal logs for dev experience
- **Configurable Log Level**: Set `LOG_LEVEL` in `.env`

### Changed
- **Switched to Ollama**: Removed OpenAI dependency — all inference runs locally
- **Improved Agent Routing**: Tighter classification prompt prevents misrouting
- **Fixed FastAPI Lifespan**: App properly initializes on startup
- **Updated LangChain Imports**: Fixed deprecated import paths
- **MCP Servers Optional**: Backend starts fine with zero MCP servers running

### Removed
- **Dead Code**: Removed `utils/Clients.py` (300+ lines unused)
- **Broken Streaming Endpoint**: Removed old non-functional `/chat/stream`
- **Hardcoded Sidebar**: Replaced static data with live backend fetch

### Fixed
- **Message Model**: Request model includes all required fields
- **Session Management**: ConversationDB uses proper context manager
- **CSS Variables**: Added all missing variable definitions
- **Frontend .env**: Corrected backend URL port

---

## [1.2.0] - 2025-08-11

### Changed
- Cleaned up client.py with proper documentation
- Updated to GPT-5 model

### Fixed
- Tavily-Langchain depreciation warning

---

## [1.1.0] - 2025-06-27

### Added
- Multi-Agent Client Framework
- Web search, RAG, and general conversation agents
- MCP Server Integration
- Example database and calendar servers

---

## [1.0.0] - 2025-06-23

### Added
- Initial release
- Modular multi-agent architecture
- PDF/image processing
- Vector search with FAISS
- Conversation history (in-memory)
