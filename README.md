# Mosaic - Multi-Agent AI Assistant

A modular multi-agent system with a Next.js frontend and FastAPI backend, powered by local LLMs via [Ollama](https://ollama.com) and extensible through MCP (Model Context Protocol) servers.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Next.js](https://img.shields.io/badge/Next.js-15-black.svg)](https://nextjs.org)
[![License](https://img.shields.io/badge/License-Non--Commercial-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/Version-2.2.0-orange.svg)](VERSION)
[![MCP](https://img.shields.io/badge/MCP-1.0+-purple.svg)](https://modelcontextprotocol.io/)

---

## Overview

**Mosaic** is a full-stack AI assistant that routes queries to specialized agents — general chat, web search, document Q&A, and any MCP tool server you connect.

- **Streaming responses** — tokens appear in real-time as the LLM generates them
- **Full authentication** — NextAuth with Google, GitHub, Microsoft OAuth + credentials
- **Conversation persistence** — PostgreSQL (production) or SQLite (dev)
- **Hot-reload MCP servers** — add/remove tool servers from the Settings UI without restarting
- **Admin panel** — system diagnostics, log viewer, config inspector, danger zone
- **Multi-LLM support** — Ollama, OpenAI, vLLM, TGI, Groq, Together — switch with one env var
- **Production-ready** — Docker Compose, Redis rate limiting, connection pooling, stateless workers

---

## Architecture

```
┌─────────────────┐         ┌──────────────────────┐         ┌───────────┐
│  Next.js (3000) │◄───────►│  FastAPI (8080)      │◄───────►│ PostgreSQL│
│  + NextAuth     │  HTTP   │  Stateless Handlers  │         │ / SQLite  │
│  + Middleware   │         │  + JWT Validation    │         └───────────┘
└─────────────────┘         └──────┬───────────────┘
                                   │         ▲
                    ┌──────────────┼─────┐   │
                    ▼              ▼     │   ▼
              ┌──────────┐  ┌────────┐  │ ┌───────┐
              │ LLM      │  │ MCP    │  │ │ Redis │
              │ Provider  │  │ Servers│  │ │(cache)│
              └──────────┘  └────────┘  │ └───────┘
              Ollama/OpenAI/  Optional   │
              vLLM/TGI/Groq             │
```

---

## Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- [Ollama](https://ollama.com) installed
- Tavily API key (for web search agent)

### 1. Setup Backend

```bash
cd Backend
python -m venv ../.venv
source ../.venv/bin/activate
pip install -r ../requirements.txt
```

### 2. Configure Environment

Copy and fill in the example files:

```bash
cp Backend/.env.example Backend/.env
cp Frontend/.env.example Frontend/.env
```

**Backend `.env`** — set your passwords and keys:
```env
TAVILY_API_KEY=your_tavily_key
ADMIN_USERNAME=admin
ADMIN_PASSWORD=your_strong_password
USER_USERNAME=user
USER_PASSWORD=your_strong_password
JWT_SECRET=generate_with_python_secrets_token_hex_32
```

**Frontend `.env`** — set the auth secret:
```env
AUTH_SECRET=generate_with_openssl_rand_base64_32
```

### 3. Pull an Ollama Model

```bash
ollama pull mistral
```

### 4. Run

```bash
# Terminal 1: Ollama
ollama serve

# Terminal 2: Backend
cd Backend && source ../.venv/bin/activate
uvicorn cifastapi_mosaic:app --reload --port 8080

# Terminal 3: Frontend
cd Frontend && npm install && npm run dev
```

Open **http://localhost:3000** — log in and start chatting.

---

## Authentication

Mosaic uses [NextAuth v5](https://authjs.dev/) for authentication with multiple providers:

| Provider | Setup |
|----------|-------|
| Credentials | Set `ADMIN_PASSWORD` and `USER_PASSWORD` in backend `.env` |
| Google | Add `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` to frontend `.env` |
| GitHub | Add `GITHUB_CLIENT_ID` / `GITHUB_CLIENT_SECRET` to frontend `.env` |
| Microsoft | Add `MICROSOFT_CLIENT_ID` / `MICROSOFT_CLIENT_SECRET` to frontend `.env` |

Providers without configured client IDs are automatically hidden from the login page. Enable any combination.

### OAuth Redirect URIs

Register these in your provider's developer console:
```
http://localhost:3000/api/auth/callback/google
http://localhost:3000/api/auth/callback/github
http://localhost:3000/api/auth/callback/microsoft-entra-id
```

### Roles

- **Admin** — full access: server management, logs, all conversations, system config
- **User** — chat, manage own conversations, add/remove MCP servers

Admin role is assigned via:
- Credentials: the `ADMIN_USERNAME` account
- OAuth: emails listed in `ADMIN_EMAILS` env var

### Security

| Protection | How |
|-----------|-----|
| Session tokens | httpOnly signed cookies — invisible to JavaScript (XSS-safe) |
| CSRF | Built into NextAuth |
| Passwords | bcrypt with salt (timing-safe) |
| Brute force | Rate limiter: 5 login attempts per 5 minutes per IP |
| Route protection | Next.js middleware blocks unauthenticated/unauthorized access |
| Open-source safe | All secrets in `.env` (gitignored), zero hardcoded values |

---

## Features

### Agents
| Agent | Description | Always Available |
|-------|-------------|:---:|
| General | Writing, coding, math, explanations, creative tasks | ✅ |
| Web | Real-time info via Tavily (news, weather, scores) | ✅ |
| RAG | Query loaded PDFs and documents | ✅ |
| Database | SQLite CRUD via MCP server | Optional |
| Calendar | Google Calendar via MCP server | Optional |
| Custom | Any MCP server you add | Optional |

### API Endpoints

| Method | Path | Access | Description |
|--------|------|--------|-------------|
| `POST` | `/auth/login` | Public | Credentials login |
| `POST` | `/auth/oauth` | Internal | Backend token for OAuth users |
| `POST` | `/auth/refresh` | Public | Refresh access token |
| `GET` | `/auth/me` | Any user | Current user info |
| `GET` | `/health` | Public | Health check |
| `POST` | `/chat` | Any user | Send message |
| `POST` | `/chat/stream` | Any user | Streaming response (SSE) |
| `GET/POST/DELETE` | `/conversations` | Any user | CRUD (own only for users) |
| `GET/POST/DELETE` | `/servers` | Any user | MCP server management |
| `POST` | `/servers/refresh` | Any user | Hot-reload servers |
| `GET` | `/admin/status` | Admin | System diagnostics |
| `GET` | `/admin/config` | Admin | Runtime config viewer |
| `GET` | `/admin/logs` | Admin | Application logs |
| `GET` | `/admin/logs/errors` | Admin | Error logs |
| `GET` | `/admin/logs/requests` | Admin | HTTP request logs |
| `DELETE` | `/admin/conversations/clear` | Admin | Wipe all conversations |

---

## Deployment

### Docker Compose (recommended for production)

```bash
# Set passwords
export POSTGRES_PASSWORD=your_secure_password

# Build and start everything
docker compose up --build -d
```

This gives you: PostgreSQL (5432) + Redis (6379) + Backend (8080) + Frontend (3000)

### LLM Provider Configuration

Switch between providers with environment variables:

| Provider | Config |
|----------|--------|
| Ollama (local) | `LLM_PROVIDER=ollama` `LLM_MODEL=mistral` |
| OpenAI | `LLM_PROVIDER=openai` `LLM_MODEL=gpt-4o` `LLM_API_KEY=sk-...` |
| vLLM/TGI | `LLM_PROVIDER=compatible` `LLM_BASE_URL=http://gpu-server:8000/v1` `LLM_MODEL=llama-3.1-8b` |
| Groq | `LLM_PROVIDER=compatible` `LLM_BASE_URL=https://api.groq.com/openai/v1` `LLM_API_KEY=gsk-...` `LLM_MODEL=llama-3.1-70b-versatile` |
| Together | `LLM_PROVIDER=compatible` `LLM_BASE_URL=https://api.together.xyz/v1` `LLM_API_KEY=...` `LLM_MODEL=meta-llama/Llama-3-70b-chat-hf` |

### Database

- **Development**: SQLite (default, zero config)
- **Production**: Set `DATABASE_URL=postgresql://user:pass@host:5432/mosaic`

### Scaling

The backend is stateless — safe to run with multiple workers:
```bash
uvicorn cifastapi_mosaic:app --workers 4 --port 8080
```

Redis is required when running multiple workers (for shared rate limiting).

---

## Logging

Rotating log files in `Backend/logs/` (auto-created, gitignored):

| File | Contents | Rotation |
|------|----------|----------|
| `mosaic.log` | Full debug trace | 10MB × 5 backups |
| `mosaic_errors.log` | Errors only | 5MB × 3 backups |
| `requests.log` | Every HTTP request with timing | 10MB × 3 backups |

View logs from the admin panel or API:
```bash
curl -H "Authorization: Bearer $TOKEN" http://localhost:8080/admin/logs?lines=50
```

---

## Adding MCP Servers

### Via UI
Go to **Servers** in the sidebar → **Add MCP Server** → enter name, URL, description.

### Via API
```bash
curl -X POST http://localhost:8080/servers \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name": "my_server", "url": "http://localhost:9000/sse", "description": "Does cool stuff"}'
```

---

## Project Structure

```
Mosaic/
├── README.md
├── CHANGELOG.md
├── LICENSE
├── VERSION
├── requirements.txt
├── docker-compose.yml            # Full stack: PG + Redis + Backend + Frontend
├── .gitignore
├── Backend/
│   ├── .env.example
│   ├── Dockerfile
│   ├── cifastapi_mosaic.py       # FastAPI app + middleware
│   ├── client.py                 # AgentRegistry + MosaicHandler (stateless)
│   ├── utils/
│   │   ├── auth.py               # JWT + bcrypt + user provider interface
│   │   ├── llm.py                # LLM provider factory (Ollama/OpenAI/compatible)
│   │   ├── rate_limiter.py       # Redis or in-memory rate limiting
│   │   ├── logger.py             # Centralized rotating logs
│   │   ├── ConversationDB.py     # SQLAlchemy (SQLite or PostgreSQL)
│   │   ├── RAGTools.py           # Document Q&A tools
│   │   └── ProcessPDF.py         # PDF/image processing
│   └── servers/
│       ├── database_server.py    # SQLite MCP server
│       └── calendar_server.py    # Google Calendar MCP server
└── Frontend/
    ├── .env.example
    ├── Dockerfile
    ├── package.json
    ├── middleware.ts              # Route protection
    ├── src/
    │   ├── auth.ts               # NextAuth v5 config (OAuth + credentials)
    │   ├── lib/auth.ts           # authFetch helper
    │   └── app/
    │       ├── page.tsx          # Chat (streaming)
    │       ├── login/page.tsx    # Login (OAuth + credentials)
    │       ├── settings/page.tsx # MCP server management
    │       ├── admin/page.tsx    # Admin panel
    │       └── chat/[id]/page.tsx
    └── ...
```

---

## License

Non-Commercial, No-Distribution License (Based on MIT). See [LICENSE](LICENSE).

---

## Acknowledgments
- [Auth.js / NextAuth](https://authjs.dev/) for authentication
- [LangChain](https://langchain.com) / [LangGraph](https://langchain-ai.github.io/langgraph/) for agent orchestration
- [Ollama](https://ollama.com) for local LLM inference
- [MCP](https://modelcontextprotocol.io/) for the tool server protocol
- [Tavily](https://tavily.com) for web search
- [Next.js](https://nextjs.org) for the frontend framework
