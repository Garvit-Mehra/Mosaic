# Deployment Guide

## Local Development

```bash
# Terminal 1: Ollama
ollama serve

# Terminal 2: Backend
cd Backend && source ../.venv/bin/activate
uvicorn cifastapi_mosaic:app --reload --port 8080

# Terminal 3: Frontend
cd Frontend && npm run dev
```

No PostgreSQL or Redis needed locally — defaults to SQLite + in-memory rate limiter.

---

## Docker Compose (Production)

```bash
export POSTGRES_PASSWORD=your_secure_password
docker compose up --build -d
```

Services: PostgreSQL (5432) + Redis (6379) + Backend (8080) + Frontend (3000)

---

## LLM Provider Configuration

Switch providers with environment variables in `Backend/.env`:

| Provider | Config |
|----------|--------|
| Ollama (local) | `LLM_PROVIDER=ollama` `LLM_MODEL=mistral` |
| OpenAI | `LLM_PROVIDER=openai` `LLM_MODEL=gpt-4o` `LLM_API_KEY=sk-...` |
| vLLM / TGI | `LLM_PROVIDER=compatible` `LLM_BASE_URL=http://gpu-server:8000/v1` `LLM_MODEL=llama-3.1-8b` |
| Groq | `LLM_PROVIDER=compatible` `LLM_BASE_URL=https://api.groq.com/openai/v1` `LLM_API_KEY=gsk-...` `LLM_MODEL=llama-3.1-70b-versatile` |
| Together | `LLM_PROVIDER=compatible` `LLM_BASE_URL=https://api.together.xyz/v1` `LLM_API_KEY=...` `LLM_MODEL=meta-llama/Llama-3-70b-chat-hf` |

---

## Database

| Environment | Config |
|-------------|--------|
| Development | `DATABASE_URL=sqlite:///conversations.db` (default) |
| Production | `DATABASE_URL=postgresql://user:pass@host:5432/mosaic` |

SQLAlchemy auto-creates tables on first run. No migrations needed.

---

## Scaling

The backend is stateless — safe to run multiple workers:

```bash
uvicorn cifastapi_mosaic:app --workers 4 --port 8080
```

Redis is required when running multiple workers (shared rate limiting).

---

## Environment Variables

See [`.env.example`](.env.example) for the full list with descriptions.

### Required
- `TAVILY_API_KEY` — web search agent
- `ADMIN_PASSWORD` / `USER_PASSWORD` — login credentials
- `JWT_SECRET` — required in production (`ENVIRONMENT=production`)

### Optional
- `DATABASE_URL` — PostgreSQL connection string
- `REDIS_URL` — Redis for rate limiting
- `LLM_PROVIDER` / `LLM_MODEL` / `LLM_BASE_URL` — LLM config
- `ALLOWED_ORIGINS` — CORS domains (comma-separated)
- `MCP_SERVERS` — JSON array of server configs

---

## HTTPS

Put behind a reverse proxy (nginx or Caddy) for TLS:

```nginx
server {
    listen 443 ssl;
    server_name yourdomain.com;

    location / {
        proxy_pass http://localhost:3000;
    }

    location /api/ {
        proxy_pass http://localhost:8080/;
    }
}
```
