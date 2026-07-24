# API Reference

> Base URL: `http://localhost:8080`  
> All endpoints (except auth and health) require `Authorization: Bearer <token>`

---

## Authentication

| Method | Path | Auth | Description |
|--------|------|:----:|-------------|
| `POST` | `/auth/register` | — | Create account |
| `GET` | `/auth/check-username/:name` | — | Check availability |
| `POST` | `/auth/login` | — | Get access + refresh tokens |
| `POST` | `/auth/refresh` | — | Refresh access token |
| `POST` | `/auth/verify` | — | Email OTP (placeholder) |
| `POST` | `/auth/oauth` | Internal | Backend token for OAuth |
| `GET` | `/auth/me` | ✓ | Current user info |

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
| `POST` | `/chat` | ✓ | Send message, get response |
| `POST` | `/chat/stream` | ✓ | Stream response via SSE |

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
| `POST` | `/conversations` | ✓ | Create conversation |
| `GET` | `/conversations` | ✓ | List (own for users, all for admin) |
| `GET` | `/conversations/:id` | ✓ | Get messages |
| `PATCH` | `/conversations/:id` | ✓ | Update title |
| `DELETE` | `/conversations/:id` | ✓ | Delete |

---

## MCP Servers

> Per-user: each user manages their own server list.

| Method | Path | Auth | Description |
|--------|------|:----:|-------------|
| `GET` | `/servers` | ✓ | List with live status |
| `POST` | `/servers` | ✓ | Add server |
| `PATCH` | `/servers/:name` | ✓ | Edit URL / description |
| `DELETE` | `/servers/:name` | ✓ | Remove |
| `GET` | `/servers/:name/tools` | ✓ | List tools |
| `POST` | `/servers/refresh` | ✓ | Re-detect all servers |

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
| `GET` | `/health` | — | `{"status": "ok"}` |
