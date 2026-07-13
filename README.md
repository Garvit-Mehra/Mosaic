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
├── Backend/
│   ├── cifastapi_mosaic.py    # FastAPI app (endpoints)
│   ├── client.py              # Mosaic agent orchestration
│   ├── utils/
│   │   ├── ConversationDB.py  # SQLite conversation storage
│   │   ├── RAGTools.py        # Document Q&A tools
│   │   └── ProcessPDF.py      # PDF/image processing
│   ├── servers/
│   │   ├── database_server.py # SQLite MCP server
│   │   └── calendar_server.py # Google Calendar MCP server
│   └── requirements.txt
├── Frontend/
│   ├── src/app/
│   │   ├── page.tsx           # Main chat (streaming)
│   │   ├── chat/[id]/page.tsx # Conversation page
│   │   ├── settings/page.tsx  # MCP server management
│   │   └── components/
│   │       └── common/SideBar.tsx
│   └── package.json
└── .gitignore
```

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
