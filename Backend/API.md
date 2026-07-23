# API Reference

Base URL: `http://localhost:8080`

All endpoints except `/auth/login`, `/auth/refresh`, and `/health` require a Bearer token in the `Authorization` header.

---

## Authentication

| Method | Path | Access | Description |
|--------|------|--------|-------------|
| `POST` | `/auth/login` | Public | Credentials login → returns access + refresh tokens |
| `POST` | `/auth/refresh` | Public | Exchange refresh token for new access token |
| `POST` | `/auth/oauth` | Internal | Backend token for OAuth users (called by NextAuth) |
| `GET` | `/auth/me` | Any user | Current user info |

### POST /auth/login

```json
// Request
{"username": "admin", "password": "..."}

// Response
{
  "access_token": "eyJ...",
  "refresh_token": "eyJ...",
  "token_type": "bearer",
  "expires_in": 86400,
  "username": "admin",
  "role": "admin"
}
```

---

## Chat

| Method | Path | Access | Description |
|--------|------|--------|-------------|
| `POST` | `/chat` | Any user | Send message, get full response |
| `POST` | `/chat/stream` | Any user | Streaming response via SSE |

### POST /chat

```json
// Request
{"message": "Hello", "conversation_id": null}

// Response
{"response": "Hi there!", "agent": "general", "conversation_id": 5}
```

### POST /chat/stream

Returns Server-Sent Events:
```
data: {"type": "agent", "agent": "general"}
data: {"type": "token", "content": "Hi"}
data: {"type": "token", "content": " there!"}
data: {"type": "done", "conversation_id": 5, "full_response": "Hi there!", "agent": "general"}
```

---

## Conversations

| Method | Path | Access | Description |
|--------|------|--------|-------------|
| `POST` | `/conversations` | Any user | Create new conversation |
| `GET` | `/conversations` | Any user | List conversations (users see own, admin sees all) |
| `GET` | `/conversations/:id` | Any user | Get messages (ownership enforced) |
| `PATCH` | `/conversations/:id` | Any user | Update title |
| `DELETE` | `/conversations/:id` | Any user | Delete conversation |

---

## MCP Servers

| Method | Path | Access | Description |
|--------|------|--------|-------------|
| `GET` | `/servers` | Any user | List all servers with status |
| `POST` | `/servers` | Any user | Add new MCP server |
| `DELETE` | `/servers/:name` | Any user | Remove server |
| `GET` | `/servers/:name/tools` | Any user | List tools for a server |
| `POST` | `/servers/refresh` | Any user | Re-detect online/offline servers |

### POST /servers

```json
// Request
{"name": "my_server", "description": "Does stuff", "url": "http://localhost:9000/sse"}

// Response
{"message": "Server 'my_server' added. Connected.", "connected": true, "server": {...}}
```

---

## Admin (requires admin role)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/admin/status` | System diagnostics |
| `GET` | `/admin/config` | Runtime configuration (secrets masked) |
| `GET` | `/admin/logs?lines=100&level=ERROR` | Application logs |
| `GET` | `/admin/logs/errors` | Error-only logs |
| `GET` | `/admin/logs/requests` | HTTP request logs |
| `GET` | `/admin/conversations` | All conversations (all users) |
| `DELETE` | `/admin/conversations/clear` | Wipe all conversations |

---

## Health

| Method | Path | Access | Description |
|--------|------|--------|-------------|
| `GET` | `/health` | Public | Returns `{"status": "ok"}` |
