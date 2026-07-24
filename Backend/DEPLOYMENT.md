# Deployment

---

## Local Development

```bash
ollama serve                                            # Terminal 1
cd Backend && source ../.venv/bin/activate
uvicorn cifastapi_mosaic:app --reload --port 8080       # Terminal 2
cd Frontend && npm run dev                              # Terminal 3
```

No PostgreSQL or Redis required — defaults to SQLite + in-memory rate limiter.

---

## Docker Compose

```bash
export POSTGRES_PASSWORD=secure_password
docker compose up --build -d
```

```mermaid
graph LR
    PG[(PostgreSQL)] --> Backend
    Redis[(Redis)] --> Backend
    Backend[FastAPI :8080] --> Frontend[Next.js :3000]
```

---

## LLM Providers

Configure in `Backend/.env`:

| Provider | Environment Variables |
|----------|---------------------|
| **Ollama** (default) | `LLM_PROVIDER=ollama` `LLM_MODEL=mistral` |
| **OpenAI** | `LLM_PROVIDER=openai` `LLM_MODEL=gpt-4o` `LLM_API_KEY=sk-...` |
| **Groq** | `LLM_PROVIDER=compatible` `LLM_BASE_URL=https://api.groq.com/openai/v1` `LLM_API_KEY=gsk-...` |
| **vLLM / TGI** | `LLM_PROVIDER=compatible` `LLM_BASE_URL=http://gpu:8000/v1` `LLM_MODEL=llama-3.1-8b` |
| **Together** | `LLM_PROVIDER=compatible` `LLM_BASE_URL=https://api.together.xyz/v1` `LLM_API_KEY=...` |

---

## Database

| Environment | Configuration |
|-------------|--------------|
| Development | `DATABASE_URL=sqlite:///conversations.db` (default) |
| Production | `DATABASE_URL=postgresql://user:pass@host:5432/mosaic` |

Tables are auto-created on first run.

---

## Scaling

```bash
uvicorn cifastapi_mosaic:app --workers 4 --port 8080
```

The backend is stateless — multiple workers share the database and Redis. Redis is required for shared rate limiting when using multiple workers.

---

## HTTPS

Use a reverse proxy:

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

---

## Environment Reference

See [`Backend/.env.example`](.env.example) and [`Frontend/.env.example`](../Frontend/.env.example) for all variables with descriptions.
