# Mosaic - Multi-Agent AI Assistant

A modular multi-agent system with a Next.js frontend and FastAPI backend, powered by local LLMs via [Ollama](https://ollama.com) and extensible through MCP (Model Context Protocol) servers.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Next.js](https://img.shields.io/badge/Next.js-15-black.svg)](https://nextjs.org)
[![License](https://img.shields.io/badge/License-Non--Commercial-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/Version-2.1.0-orange.svg)](VERSION)
[![MCP](https://img.shields.io/badge/MCP-1.0+-purple.svg)](https://modelcontextprotocol.io/)

---

## Overview

**Mosaic** is a full-stack AI assistant that routes queries to specialized agents — general chat, web search, document Q&A, and any MCP tool server you connect.

- **Streaming responses** — tokens appear in real-time as the LLM generates them
- **Full authentication** — NextAuth with Google, GitHub, Microsoft OAuth + credentials
- **Conversation persistence** — chat history stored in SQLite, survives restarts
- **Hot-reload MCP servers** — add/remove tool servers from the Settings UI without restarting
- **Admin panel** — system diagnostics, log viewer, config inspector, danger zone
- **Local-first** — runs entirely on your machine with Ollama

---

## Architecture

```
┌─────────────────┐         ┌──────────────────────┐
│  Next.js (3000) │◄───────►│  FastAPI (8080)      │
│  + NextAuth     │  HTTP   │  Agent Orchestration │
│  + Middleware   │         │  + JWT Validation    │
└─────────────────┘         └──────┬───────────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼              ▼
              ┌──────────┐  ┌──────────┐  ┌──────────┐
              │ Ollama   │  │ DB Server│  │ Calendar │
              │ (LLM)   │  │ (8000)   │  │ (8010)   │
              └──────────┘  └──────────┘  └──────────┘
                              MCP Servers (optional)
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
├── requirements.txt
├── .gitignore
├── Backend/
│   ├── .env.example
│   ├── cifastapi_mosaic.py       # FastAPI app + middleware
│   ├── client.py                 # Agent orchestration
│   ├── utils/
│   │   ├── auth.py               # JWT + bcrypt + rate limiting
│   │   ├── logger.py             # Centralized logging
│   │   ├── ConversationDB.py     # SQLite persistence
│   │   ├── RAGTools.py           # Document Q&A
│   │   └── ProcessPDF.py         # PDF/image processing
│   └── servers/
│       ├── database_server.py    # SQLite MCP server
│       └── calendar_server.py    # Google Calendar MCP server
└── Frontend/
    ├── .env.example
    ├── package.json
    ├── middleware.ts              # Route protection
    ├── src/
    │   ├── auth.ts               # NextAuth v5 config
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
