# API Reference

> Base URL: `http://localhost:8080`  
> All endpoints (except auth and health) require `Authorization: Bearer <token>`

---

## Authentication

| Method | Path | Auth | Description |
|--------|------|:----:|-------------|
| `POST` | `/auth/register` | No | Create account |
| `GET` | `/auth/check-username/:name` | No | Check availability |
| `POST` | `/auth/login` | No | Get access + refresh tokens |
| `POST` | `/auth/refresh` | No | Refresh access token |
| `POST` | `/auth/verify` | No | Email OTP (placeholder) |
| `POST` | `/auth/oauth` | Internal | Backend token for OAuth |
| `GET` | `/auth/me` | Yes | Current user info |

<details>
<summary><strong>POST /auth/login</strong></summary>

```json
// Request
{ "username": "admin", "password": "..." }

// Response 200
{
  "access_token": "eyJ...",
  "refresh_token": "eyJ...",
  "token_type": "bearer",
  "expires_in": 86400,
  "username": "admin",
  "role": "admin"
}
```

</details>

---

## Chat

| Method | Path | Auth | Description |
|--------|------|:----:|-------------|
| `POST` | `/chat` | Yes | Send message, get response |
| `POST` | `/chat/stream` | Yes | Stream response via SSE |

<details>
<summary><strong>POST /chat</strong></summary>

```json
// Request
{ "message": "Hello", "conversation_id": null }

// Response 200
{ "response": "Hi!", "agent": "general", "conversation_id": 5 }
```

</details>

<details>
<summary><strong>POST /chat/stream — SSE Events</strong></summary>

```
data: {"type": "agent", "agent": "general"}
data: {"type": "token", "content": "Hi"}
data: {"type": "token", "content": "!"}
data: {"type": "done", "conversation_id": 5, "full_response": "Hi!", "agent": "general"}
```

</details>

---

## Conversations

| Method | Path | Auth | Description |
|--------|------|:----:|-------------|
| `POST` | `/conversations` | Yes | Create conversation |
| `GET` | `/conversations` | Yes | List (own for users, all for admin) |
| `GET` | `/conversations/:id` | Yes | Get messages |
| `PATCH` | `/conversations/:id` | Yes | Update title |
| `DELETE` | `/conversations/:id` | Yes | Delete |

---

## MCP Servers

> Per-user: each user manages their own server list.

| Method | Path | Auth | Description |
|--------|------|:----:|-------------|
| `GET` | `/servers` | Yes | List with live status |
| `POST` | `/servers` | Yes | Add server |
| `PATCH` | `/servers/:name` | Yes | Edit URL / description |
| `DELETE` | `/servers/:name` | Yes | Remove |
| `GET` | `/servers/:name/tools` | Yes | List tools |
| `POST` | `/servers/refresh` | Yes | Re-detect all servers |

<details>
<summary><strong>POST /servers</strong></summary>

```json
// Request
{
  "name": "crypto",
  "description": "Crypto whale insights",
  "url": "https://cryptowhaleinsights.com/mcp",
  "transport": "streamable_http"  // optional, auto-detects
}

// Response 200
{ "message": "Server 'crypto' added. Connected.", "connected": true }
```

</details>

---

## Admin

> Requires `role: "admin"`

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/admin/status` | System diagnostics |
| `GET` | `/admin/config` | Runtime configuration |
| `GET` | `/admin/logs` | Application logs |
| `GET` | `/admin/logs/errors` | Error logs |
| `GET` | `/admin/logs/requests` | Request logs |
| `GET` | `/admin/conversations` | All conversations |
| `DELETE` | `/admin/conversations/clear` | Wipe all |

---

## Health

| Method | Path | Auth | Description |
|--------|------|:----:|-------------|
| `GET` | `/health` | No | `{"status": "ok"}` |
