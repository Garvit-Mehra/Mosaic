# Mosaic - Multi-Agent AI Assistant

A modular multi-agent system with a Next.js frontend and FastAPI backend, powered by local LLMs via [Ollama](https://ollama.com) and extensible through MCP (Model Context Protocol) servers.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Next.js](https://img.shields.io/badge/Next.js-15-black.svg)](https://nextjs.org)
[![License](https://img.shields.io/badge/License-Non--Commercial-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/Version-2.0.0-orange.svg)](VERSION)
[![MCP](https://img.shields.io/badge/MCP-1.0+-purple.svg)](https://modelcontextprotocol.io/)

---

## Overview

**Mosaic** is a full-stack AI assistant that routes queries to specialized agents — general chat, web search, document Q&A, and any MCP tool server you connect.

- **Streaming responses** — tokens appear in real-time as the LLM generates them
- **Conversation persistence** — chat history stored in SQLite, survives restarts
- **Hot-reload MCP servers** — add/remove tool servers from the Settings UI without restarting
- **Local-first** — runs entirely on your machine with Ollama, no API keys required (except Tavily for web search)

---

## Architecture

```
┌─────────────────┐         ┌──────────────────────┐
│  Next.js (3000) │◄───────►│  FastAPI (8080)      │
│  Frontend       │  HTTP   │  Agent Orchestration │
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
pip install -r requirements.txt
```

### 2. Configure Environment

Create `Backend/.env`:
```env
TAVILY_API_KEY=your_tavily_key_here
LOG_LEVEL=INFO   # DEBUG, INFO, WARNING, or ERROR
```

Create `Frontend/.env`:
```env
NEXT_PUBLIC_BACKEND_URL=http://127.0.0.1:8080
```

### 3. Pull an Ollama Model

```bash
ollama pull mistral
```

(Or use `llama3.2` for faster responses — change model in `cifastapi_mosaic.py`)

### 4. Run

```bash
# Terminal 1: Ollama (if not already running as a service)
ollama serve

# Terminal 2: Backend
cd Backend
source ../.venv/bin/activate
uvicorn cifastapi_mosaic:app --reload --port 8080

# Terminal 3: Frontend
cd Frontend
npm install
npm run dev
```

Open **http://localhost:3000** — start chatting.

### 5. (Optional) MCP Servers

```bash
# Terminal 4: Database tools
cd Backend
python servers/database_server.py

# Terminal 5: Google Calendar tools
python servers/calendar_server.py
```

Then go to **Settings** in the UI and click **Refresh All**, or add new servers via the form.

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

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/chat` | Send message, get response |
| `POST` | `/chat/stream` | Streaming response (SSE) |
| `GET` | `/conversations` | List all conversations |
| `GET` | `/conversations/:id` | Get conversation messages |
| `DELETE` | `/conversations/:id` | Delete conversation |
| `GET` | `/servers` | List MCP servers + status |
| `POST` | `/servers` | Add new MCP server |
| `DELETE` | `/servers/:name` | Remove MCP server |
| `POST` | `/servers/refresh` | Hot-reload all servers |
| `GET` | `/servers/:name/tools` | List tools for a server |
| `GET` | `/logs?lines=100&level=ERROR` | View recent logs |
| `GET` | `/logs/errors` | View error-only logs |
| `GET` | `/status` | Active agents and server status |

---

## Adding MCP Servers

### Via UI
Go to **Settings** → **Add MCP Server** → enter name, URL, description → click Add.

### Via API
```bash
curl -X POST http://localhost:8080/servers \
  -H "Content-Type: application/json" \
  -d '{"name": "my_server", "url": "http://localhost:9000/sse", "description": "Does cool stuff"}'
```

### Via Config
Edit `SERVER_CONFIGS` in `cifastapi_mosaic.py`:
```python
SERVER_CONFIGS = [
    {"name": "my_server", "description": "...", "url": "http://localhost:9000/sse"},
]
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
│   ├── .gitignore
│   ├── cifastapi_mosaic.py    # FastAPI app (endpoints + middleware)
│   ├── client.py              # Mosaic agent orchestration
│   ├── logs/                  # Auto-created log files (gitignored)
│   │   ├── mosaic.log         # Full debug log (rotating, 10MB)
│   │   ├── mosaic_errors.log  # Errors only (rotating, 5MB)
│   │   └── requests.log       # HTTP request log (rotating, 10MB)
│   ├── utils/
│   │   ├── logger.py          # Centralized logging configuration
│   │   ├── ConversationDB.py  # SQLite conversation storage
│   │   ├── RAGTools.py        # Document Q&A tools
│   │   └── ProcessPDF.py      # PDF/image processing
│   ├── servers/
│   │   ├── database_server.py # SQLite MCP server
│   │   └── calendar_server.py # Google Calendar MCP server
│   └── examples/
│       └── mosaic_template.py
└── Frontend/
    ├── .gitignore
    ├── package.json
    ├── src/app/
    │   ├── page.tsx           # Main chat (streaming)
    │   ├── chat/[id]/page.tsx # Conversation page
    │   ├── settings/page.tsx  # MCP server management
    │   └── components/
    │       └── common/SideBar.tsx
    └── ...
```

---

## Logging

Mosaic has a centralized logging system that writes to rotating log files in `Backend/logs/`.

| File | What it captures | Size limit |
|------|-----------------|-----------|
| `mosaic.log` | Everything (DEBUG+): agent routing, MCP connections, errors | 10MB × 5 |
| `mosaic_errors.log` | Errors only — check this first when debugging | 5MB × 3 |
| `requests.log` | Every HTTP request: method, path, status, duration, body | 10MB × 3 |

**View logs from the API:**
```bash
# Recent 50 lines
curl http://localhost:8080/logs?lines=50

# Errors only
curl http://localhost:8080/logs/errors

# Filter by level
curl http://localhost:8080/logs?level=WARNING
```

**Set log level** in `Backend/.env`:
```env
LOG_LEVEL=DEBUG  # DEBUG, INFO, WARNING, ERROR
```

**Console output** shows colored logs (green=info, yellow=warning, red=error) during development.

---

## License

Non-Commercial, No-Distribution License (Based on MIT). See [LICENSE](LICENSE).

---

## Acknowledgments
- [LangChain](https://langchain.com) / [LangGraph](https://langchain-ai.github.io/langgraph/) for agent orchestration
- [Ollama](https://ollama.com) for local LLM inference
- [MCP](https://modelcontextprotocol.io/) for the tool server protocol
- [Tavily](https://tavily.com) for web search
- [Next.js](https://nextjs.org) for the frontend framework
