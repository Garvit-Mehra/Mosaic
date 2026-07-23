# Authentication

Mosaic uses [NextAuth v5 (Auth.js)](https://authjs.dev/) for authentication.

---

## Providers

| Provider | How to enable |
|----------|---------------|
| Credentials | Set `ADMIN_PASSWORD` and `USER_PASSWORD` in `Backend/.env` |
| Google | Set `GOOGLE_CLIENT_ID` + `GOOGLE_CLIENT_SECRET` in `Frontend/.env` |
| GitHub | Set `GITHUB_CLIENT_ID` + `GITHUB_CLIENT_SECRET` in `Frontend/.env` |
| Microsoft | Set `MICROSOFT_CLIENT_ID` + `MICROSOFT_CLIENT_SECRET` in `Frontend/.env` |

Providers without configured client IDs are automatically hidden from the login page.

---

## OAuth Redirect URIs

Register in your provider's developer console:

```
http://localhost:3000/api/auth/callback/google
http://localhost:3000/api/auth/callback/github
http://localhost:3000/api/auth/callback/microsoft-entra-id
```

Replace `localhost:3000` with your domain in production.

---

## Roles

| Role | Assigned via | Access |
|------|-------------|--------|
| `admin` | Credentials: `ADMIN_USERNAME` account | Everything |
| `admin` | OAuth: email in `ADMIN_EMAILS` env var | Everything |
| `user` | All other logins | Chat, own conversations, MCP servers |

---

## Session Flow

1. User logs in (credentials or OAuth)
2. NextAuth creates a signed httpOnly cookie (JWT session)
3. Next.js middleware validates the cookie on every page load
4. `authFetch()` extracts the backend token from the session and passes it to the API

---

## Security

| Concern | Protection |
|---------|-----------|
| XSS token theft | Sessions in httpOnly cookies — JS can't access |
| CSRF | Built into NextAuth |
| Brute force | Rate limiter (5 attempts / 5 min per IP) |
| Password storage | bcrypt with salt |
| Token forgery | Signed with `AUTH_SECRET` (frontend) + `JWT_SECRET` (backend) |
| Route protection | Next.js middleware blocks unauthenticated access server-side |

---

## Environment Variables

In `Frontend/.env`:

```env
AUTH_SECRET=openssl_rand_base64_32
AUTH_TRUST_HOST=true

# OAuth (all optional)
GOOGLE_CLIENT_ID=...
GOOGLE_CLIENT_SECRET=...
GITHUB_CLIENT_ID=...
GITHUB_CLIENT_SECRET=...
MICROSOFT_CLIENT_ID=...
MICROSOFT_CLIENT_SECRET=...
MICROSOFT_TENANT_ID=common

# Admin emails for OAuth users
ADMIN_EMAILS=admin@yourdomain.com,another@admin.com
```
