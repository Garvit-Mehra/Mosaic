# Mosaic

A modular multi-agent AI assistant with streaming chat, MCP tool servers, and multi-provider LLM support.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Next.js](https://img.shields.io/badge/Next.js-15-black.svg)](https://nextjs.org)
[![License](https://img.shields.io/badge/License-Non--Commercial-green.svg)](LICENSE)

---

## What it does

- Routes queries to specialized agents (general, web search, RAG, custom MCP tools)
- Streams responses token-by-token
- Persists conversations across sessions
- Connects to any MCP server for extensible tooling
- Supports Ollama, OpenAI, Groq, vLLM, TGI, and any OpenAI-compatible API

---

## Quick Start

```bash
# Clone and setup (checks all dependencies, installs everything)
git clone https://github.com/Garvit-Mehra/Mosaic.git
cd Mosaic
./setup.sh

# Start all services
./start.sh

# Stop all services
./start.sh stop
```

Or manually:

```bash
# 1. Backend
cd Backend
python -m venv ../.venv && source ../.venv/bin/activate
pip install -r ../requirements.txt
cp .env.example .env  # fill in your keys

# 2. Frontend
cd Frontend
npm install
cp .env.example .env  # set AUTH_SECRET

# 3. Pull a model
ollama pull mistral

# 4. Run
ollama serve                                              # Terminal 1
cd Backend && uvicorn cifastapi_mosaic:app --port 8080    # Terminal 2
cd Frontend && npm run dev                                # Terminal 3
```

Open `http://localhost:3000`

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Next.js 15, React 19, Tailwind CSS, NextAuth v5 |
| Backend | FastAPI, LangChain, LangGraph, SQLAlchemy |
| LLM | Ollama / OpenAI / any OpenAI-compatible API |
| Database | SQLite (dev) or PostgreSQL (prod) |
| Cache | Redis (optional, for rate limiting) |
| Auth | NextAuth + JWT + bcrypt |
| Infra | Docker Compose |

---

## Documentation

| Topic | Location |
|-------|----------|
| API endpoints | [`Backend/API.md`](Backend/API.md) |
| Deployment & scaling | [`Backend/DEPLOYMENT.md`](Backend/DEPLOYMENT.md) |
| Authentication & OAuth | [`Frontend/AUTH.md`](Frontend/AUTH.md) |
| Changelog | [`CHANGELOG.md`](CHANGELOG.md) |
| Backend env vars | [`Backend/.env.example`](Backend/.env.example) |
| Frontend env vars | [`Frontend/.env.example`](Frontend/.env.example) |

---

## Docker

```bash
docker compose up --build
```

Runs PostgreSQL + Redis + Backend + Frontend. See [`Backend/DEPLOYMENT.md`](Backend/DEPLOYMENT.md) for details.

---

## License

Non-Commercial, No-Distribution License (Based on MIT). See [LICENSE](LICENSE).
