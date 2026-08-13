from fastapi import FastAPI, HTTPException, Request, Depends
from client import AgentRegistry, MosaicHandler, is_server_active
from pydantic import BaseModel, Field
from typing import Optional, List
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from utils.ConversationDB import ConversationManager
from utils.UserDB import UserManager
from utils.logger import get_logger, get_request_logger
from utils.rate_limiter import create_rate_limiter
from utils.auth import (
    LoginRequest, TokenUser,
    get_current_user, require_admin,
    authenticate_user, create_access_token, create_refresh_token,
    verify_token,
)
import json
import time
import os
import ipaddress
from urllib.parse import urlparse

from dotenv import load_dotenv
load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OAUTH_SHARED_SECRET = os.getenv("OAUTH_SHARED_SECRET", "")
MAX_REQUEST_BODY_BYTES = int(os.getenv("MAX_REQUEST_BODY_BYTES", str(1 * 1024 * 1024)))  # 1MB default

logger = get_logger("api")
request_logger = get_request_logger()

# --- Configuration from environment ---
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# Start with no global MCP servers — each user has their own
SERVER_CONFIGS: list = []

# Initialize core services
conversation_db = ConversationManager()
registry = AgentRegistry()
handler: Optional[MosaicHandler] = None
rate_limiter = create_rate_limiter(
    max_attempts=int(os.getenv("LOGIN_RATE_LIMIT", "5")),
    window_seconds=int(os.getenv("LOGIN_RATE_WINDOW", "300")),
)
chat_rate_limiter = create_rate_limiter(
    max_attempts=int(os.getenv("CHAT_RATE_LIMIT", "20")),
    window_seconds=int(os.getenv("CHAT_RATE_WINDOW", "60")),
)
register_rate_limiter = create_rate_limiter(
    max_attempts=int(os.getenv("REGISTER_RATE_LIMIT", "3")),
    window_seconds=int(os.getenv("REGISTER_RATE_WINDOW", "3600")),  # 3 per hour per IP
)
STREAM_TIMEOUT = int(os.getenv("STREAM_TIMEOUT_SECONDS", "120"))


# --- Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global handler
    logger.info("Mosaic backend starting up...")
    await registry.initialize(SERVER_CONFIGS, web_search=bool(TAVILY_API_KEY))
    handler = MosaicHandler(registry, conversation_db)
    logger.info(f"✓ Agents loaded: {[a['name'] for a in registry.agents]}")
    if registry.inactive_servers:
        logger.warning(f"⚠ Inactive MCP servers: {registry.inactive_servers}")
    logger.info("✓ Backend ready — accepting requests")
    yield
    logger.info("Mosaic backend shutting down...")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount FlowQ as a sub-application at /jobs
try:
    from flowq.src.main import create_app as create_flowq_app
    flowq_app = create_flowq_app()
    app.mount("/flowq", flowq_app)
    logger.info("FlowQ mounted at /flowq")
except ImportError as e:
    logger.warning(f"FlowQ not available (missing dependencies): {e}")
except Exception as e:
    logger.warning(f"FlowQ failed to initialize: {e}")


# --- Request body size limit middleware ---
@app.middleware("http")
async def limit_request_body(request: Request, call_next):
    if request.method in ("POST", "PUT", "PATCH"):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_REQUEST_BODY_BYTES:
            return StreamingResponse(
                iter([json.dumps({"detail": f"Request body too large. Max {MAX_REQUEST_BODY_BYTES // 1024}KB."})]),
                status_code=413,
                media_type="application/json",
            )
    return await call_next(request)


# --- Request logging middleware ---
@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Skip logging for OPTIONS preflight requests
    if request.method == "OPTIONS":
        return await call_next(request)

    start_time = time.time()

    body = None
    if request.method in ("POST", "PUT", "PATCH"):
        try:
            body_bytes = await request.body()
            body = body_bytes.decode("utf-8")[:500]
        except Exception:
            body = "<unreadable>"

    response = await call_next(request)

    duration_ms = (time.time() - start_time) * 1000

    log_line = f"{request.method} {request.url.path} → {response.status_code} ({duration_ms:.0f}ms)"

    if response.status_code >= 500:
        logger.error(log_line)
    elif response.status_code >= 400:
        logger.warning(log_line)

    request_logger.info(
        f"{request.method} {request.url.path} | "
        f"status={response.status_code} | "
        f"duration={duration_ms:.0f}ms | "
        f"body={body if body else '-'}"
    )

    return response


# --- Global exception handler (no tracebacks to client) ---
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.method} {request.url.path}: {type(exc).__name__}: {exc}")
    return StreamingResponse(
        iter([json.dumps({"detail": "An internal error occurred. Please try again."})]),
        status_code=500,
        media_type="application/json",
    )


# =============================================================================
# PUBLIC ENDPOINTS (no auth required)
# =============================================================================

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    conversation_id: Optional[int] = None
    transient: bool = False


class ConversationCreateRequest(BaseModel):
    title: str


class ConversationUpdateRequest(BaseModel):
    title: str


# --- Auth ---
class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., min_length=5, max_length=255)
    password: str = Field(..., min_length=6, max_length=128)


@app.post("/auth/register")
async def register(req: RegisterRequest, request: Request):
    """
    Register a new user account.
    Rate limited: 3 registrations per hour per IP.
    """
    client_ip = request.client.host if request.client else "unknown"

    if not register_rate_limiter.check(f"register:{client_ip}"):
        raise HTTPException(
            status_code=429,
            detail="Too many registration attempts. Try again later.",
            headers={"Retry-After": "3600"},
        )

    try:
        user = user_db.create_user(
            username=req.username,
            email=req.email,
            password=req.password,
            role="user",
            verified=False,  # Will be set to True after email OTP verification
        )
        logger.info(f"New user registered: {req.username} ({req.email})")
        return {
            "message": "Account created. Please verify your email.",
            "username": user["username"],
            "email": user["email"],
            "verified": user["verified"],
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/auth/check-username/{username}")
async def check_username(username: str):
    """Check if a username is available. No auth required."""
    import re
    if not re.match(r'^[a-zA-Z0-9_]{3,50}$', username):
        return {"available": False, "reason": "Only letters, numbers, and underscores (3-50 chars)."}

    user = user_db.get_user_by_username(username)
    if user:
        return {"available": False, "reason": "Username is taken."}
    return {"available": True}


@app.post("/auth/verify")
async def verify_email(username: str = "", otp: str = ""):
    """
    Placeholder for email OTP verification.
    Currently marks the user as verified without actual OTP check.
    TODO: Implement actual email sending + OTP validation.
    """
    # PLACEHOLDER: In production, validate the OTP code sent to email
    # For now, this is a no-op endpoint that the frontend calls
    return {"message": "Verification endpoint placeholder. Not yet implemented."}


@app.post("/auth/login")
async def login(req: LoginRequest, request: Request):
    """
    Authenticate with username/password. Returns access + refresh tokens.
    Rate limited: 5 attempts per 5 minutes per IP.
    """
    client_ip = request.client.host if request.client else "unknown"

    # Rate limiting
    if not rate_limiter.check(client_ip):
        logger.warning(f"Rate limited login from IP: {client_ip}")
        raise HTTPException(
            status_code=429,
            detail="Too many login attempts. Try again later.",
            headers={"Retry-After": "300"},
        )

    user = authenticate_user(req.username, req.password)
    if not user:
        remaining = rate_limiter.remaining(client_ip)
        logger.warning(f"Failed login: username={req.username} ip={client_ip} remaining={remaining}")
        raise HTTPException(status_code=401, detail="Invalid username or password.")
        
    if user.get("verified") is False:
        logger.warning(f"Login denied: Unverified account {req.username} ip={client_ip}")
        raise HTTPException(status_code=403, detail="Account not verified. Please check your email.")

    access_token = create_access_token(user["username"], user["role"])
    refresh_token = create_refresh_token(user["username"], user["role"])

    logger.info(f"Login success: {user['username']} (role={user['role']}) ip={client_ip}")

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": 24 * 3600,
        "username": user["username"],
        "role": user["role"],
    }


@app.post("/auth/refresh")
async def refresh_token_endpoint(request: Request):
    """
    Get a new access token using a valid refresh token.
    Send refresh token in Authorization header.
    """
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Refresh token required.")

    token = auth_header[7:]
    user = verify_token(token, expected_type="refresh")

    new_access = create_access_token(user.username, user.role)
    return {
        "access_token": new_access,
        "token_type": "bearer",
        "expires_in": 24 * 3600,
    }


@app.get("/auth/me")
async def get_me(user: TokenUser = Depends(get_current_user)):
    """Get the current authenticated user's info."""
    return {"username": user.username, "role": user.role}


@app.get("/health")
async def health():
    """Health check — no auth required. Use for uptime monitoring."""
    return {"status": "ok", "timestamp": int(time.time())}


class OAuthRequest(BaseModel):
    email: str
    name: Optional[str] = None
    provider: str
    role: str = "user"


@app.post("/auth/oauth")
async def oauth_login(req: OAuthRequest, request: Request):
    """
    Called by the NextAuth callback to get a backend access token for OAuth users.
    Protected by a shared secret to prevent external abuse.
    """
    # Verify shared secret (set in both Backend/.env and Frontend/.env)
    if not OAUTH_SHARED_SECRET:
        logger.error(f"OAuth login attempted but OAUTH_SHARED_SECRET is not configured. Denying access.")
        raise HTTPException(status_code=500, detail="OAuth not configured on this server.")

    provided_secret = request.headers.get("X-OAuth-Secret", "")
    if provided_secret != OAUTH_SHARED_SECRET:
        logger.warning(f"OAuth endpoint called with invalid secret from {request.client.host if request.client else 'unknown'}")
        raise HTTPException(status_code=403, detail="Forbidden.")

    access_token = create_access_token(req.email, req.role)
    logger.info(f"OAuth login: {req.email} via {req.provider} (role={req.role})")
    return {"access_token": access_token}


# =============================================================================
# USER ENDPOINTS (any authenticated user)
# =============================================================================

# --- Chat ---
@app.post("/chat")
async def chat(req: ChatRequest, user: TokenUser = Depends(get_current_user)):
    """Send a message and get a response."""
    # Rate limit per user
    if not chat_rate_limiter.check(f"chat:{user.username}"):
        raise HTTPException(status_code=429, detail="Too many messages. Please slow down.")

    conv_id = req.conversation_id

    # If conversation_id provided, verify ownership
    if conv_id is not None:
        convo = conversation_db.get_conversation(conv_id)
        if not convo:
            raise HTTPException(status_code=404, detail="Conversation not found")
        if user.role != "admin" and getattr(convo, 'user_id', None) != user.username:
            raise HTTPException(status_code=403, detail="Access denied.")
    elif req.transient:
        import uuid
        conv_id = f"transient-{uuid.uuid4()}"
    else:
        title = req.message[:50] + ("..." if len(req.message) > 50 else "")
        convo = conversation_db.create_conversation(title=title, user_id=user.username)
        conv_id = convo.id
        logger.info(f"Created conversation id={conv_id} for user={user.username}")

    logger.info(f"Chat: user={user.username} conv={conv_id} msg='{req.message[:60]}'")

    result = await handler.chat(
        req.message,
        conversation_id=conv_id,
        user_id=user.username,
    )

    logger.info(f"Response: conv={conv_id} agent={result.get('agent')} len={len(result['response'])}")

    if not req.transient:
        conversation_db.add_message(conv_id, "user", req.message)
        conversation_db.add_message(conv_id, "assistant", result["response"], agent=result.get("agent"))

    return {
        "response": result["response"],
        "agent": result.get("agent"),
        "conversation_id": conv_id,
    }


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest, user: TokenUser = Depends(get_current_user)):
    """Stream a response via Server-Sent Events."""
    import asyncio

    # Rate limit per user
    if not chat_rate_limiter.check(f"chat:{user.username}"):
        raise HTTPException(status_code=429, detail="Too many messages. Please slow down.")

    conv_id = req.conversation_id

    # Verify ownership if conversation_id provided
    if conv_id is not None:
        convo = conversation_db.get_conversation(conv_id)
        if not convo:
            raise HTTPException(status_code=404, detail="Conversation not found")
        if user.role != "admin" and getattr(convo, 'user_id', None) != user.username:
            raise HTTPException(status_code=403, detail="Access denied.")
    elif req.transient:
        import uuid
        conv_id = f"transient-{uuid.uuid4()}"
    else:
        title = req.message[:50] + ("..." if len(req.message) > 50 else "")
        convo = conversation_db.create_conversation(title=title, user_id=user.username)
        conv_id = convo.id

    logger.info(f"[stream] user={user.username} conv={conv_id} msg='{req.message[:60]}'")

    async def event_generator():
        full_response = ""
        agent_name = None
        token_count = 0
        start_time = time.time()

        try:
            async for chunk in handler.chat_stream(
                req.message,
                conversation_id=conv_id,
                user_id=user.username,
            ):
                # Timeout check
                if time.time() - start_time > STREAM_TIMEOUT:
                    yield f"data: {json.dumps({'type': 'error', 'content': 'Response timed out.'})}\n\n"
                    break

                if chunk["type"] == "agent":
                    agent_name = chunk["agent"]
                elif chunk["type"] == "token":
                    token_count += 1
                elif chunk["type"] == "done":
                    full_response = chunk.get("full_response", "")

                yield f"data: {json.dumps(chunk)}\n\n"

        except asyncio.TimeoutError:
            yield f"data: {json.dumps({'type': 'error', 'content': 'Response timed out.'})}\n\n"
        except Exception as e:
            logger.error(f"[stream] Error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': 'An error occurred.'})}\n\n"

        # Persist messages
        if not req.transient:
            conversation_db.add_message(conv_id, "user", req.message)
            if full_response:
                conversation_db.add_message(conv_id, "assistant", full_response, agent=agent_name)

        logger.info(f"[stream] Done: conv={conv_id} agent={agent_name} tokens={token_count}")
        yield f"data: {json.dumps({'type': 'done', 'conversation_id': conv_id, 'full_response': full_response, 'agent': agent_name})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


# --- Conversations (user sees only their own) ---
@app.post("/conversations")
async def create_conversation(req: ConversationCreateRequest, user: TokenUser = Depends(get_current_user)):
    """Create a new conversation."""
    convo = conversation_db.create_conversation(title=req.title, user_id=user.username)
    return {"id": convo.id, "title": convo.title, "created_at": str(convo.created_at)}


@app.get("/conversations")
async def list_conversations(user: TokenUser = Depends(get_current_user), limit: int = 50):
    """List conversations for the current user (admin sees all)."""
    user_filter = None if user.role == "admin" else user.username
    convos = conversation_db.get_conversations(user_id=user_filter, limit=limit)
    return [
        {"id": c.id, "title": c.title, "created_at": str(c.created_at), "updated_at": str(c.updated_at)}
        for c in convos
    ]


@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: int, user: TokenUser = Depends(get_current_user)):
    """Get messages for a conversation. Users can only see their own."""
    convo = conversation_db.get_conversation(conversation_id)
    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Enforce ownership (admin can see all)
    if user.role != "admin" and getattr(convo, 'user_id', None) != user.username:
        raise HTTPException(status_code=403, detail="Access denied.")

    messages = conversation_db.get_messages(conversation_id)
    return {
        "id": convo.id,
        "title": convo.title,
        "messages": [
            {"id": m.id, "role": m.role, "content": m.content, "agent": m.agent, "timestamp": str(m.timestamp)}
            for m in messages
        ],
    }


@app.patch("/conversations/{conversation_id}")
async def update_conversation(conversation_id: int, req: ConversationUpdateRequest, user: TokenUser = Depends(get_current_user)):
    """Update a conversation title. Users can only update their own."""
    convo = conversation_db.get_conversation(conversation_id)
    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found")
    if user.role != "admin" and getattr(convo, 'user_id', None) != user.username:
        raise HTTPException(status_code=403, detail="Access denied.")

    conversation_db.update_conversation_title(conversation_id, req.title)
    return {"message": "Conversation updated"}


@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: int, user: TokenUser = Depends(get_current_user)):
    """Delete a conversation. Users can only delete their own."""
    convo = conversation_db.get_conversation(conversation_id)
    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found")
    if user.role != "admin" and getattr(convo, 'user_id', None) != user.username:
        raise HTTPException(status_code=403, detail="Access denied.")

    conversation_db.delete_conversation(conversation_id)
    return {"message": "Conversation deleted"}


# =============================================================================
# USER ENDPOINTS — MCP Servers (any authenticated user)
# =============================================================================

class AddServerRequest(BaseModel):
    name: str
    description: str
    url: str
    transport: Optional[str] = None


class EditServerRequest(BaseModel):
    description: Optional[str] = None
    url: Optional[str] = None


def validate_server_url(url: str):
    """
    Validate MCP server URL.
    """
    if not url.startswith("http://") and not url.startswith("https://"):
        raise HTTPException(status_code=400, detail="Server URL must start with http:// or https://")

    parsed = urlparse(url)
    hostname = parsed.hostname

    if not hostname:
        raise HTTPException(status_code=400, detail="Invalid URL: no hostname.")

    # Note: SSRF protection (blocking localhost/private IPs) has been temporarily
    # disabled so that local development servers can be added.
    
    # Resolve and check IP ranges
    try:
        import socket
        resolved = socket.getaddrinfo(hostname, None)
        for _, _, _, _, addr in resolved:
            ip = ipaddress.ip_address(addr[0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                # raise HTTPException(
                #     status_code=400,
                #     detail="Server URL resolves to a private/internal IP address. This is not allowed."
                # )
                pass
    except socket.gaierror:
        # Can't resolve — allow it (might be a hostname only reachable from certain networks)
        pass
    except HTTPException:
        raise
    except Exception:
        pass


@app.get("/servers")
async def list_servers(user: TokenUser = Depends(get_current_user)):
    """List MCP servers for the current user with live status."""
    user_configs = user_db.get_user_servers(user.username)

    servers = []
    for config in user_configs:
        active = await is_server_active(config["url"])
        agent_loaded = any(a["name"] == config["name"] for a in registry.agents)
        servers.append({
            "name": config["name"],
            "description": config["description"],
            "url": config["url"],
            "transport": config.get("transport"),
            "active": active,
            "agent_loaded": agent_loaded,
        })
    return {"servers": servers}


@app.post("/servers")
async def add_server(req: AddServerRequest, user: TokenUser = Depends(get_current_user)):
    """Add a new MCP server for the current user and try to connect."""
    validate_server_url(req.url)

    try:
        user_db.add_user_server(user.username, req.name, req.description, req.url, req.transport)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Add to registry and try to connect
    new_config = {"name": req.name, "description": req.description, "url": req.url}
    if req.transport:
        new_config["transport"] = req.transport

    logger.info(f"[{user.username}] Adding MCP server: {req.name} @ {req.url}")

    if new_config["name"] not in [c["name"] for c in registry.server_configs]:
        registry.server_configs.append(new_config)
        registry.inactive_servers.append(req.name)

    result = await registry.refresh_mcp_servers()
    connected = req.name in result.get("connected", [])

    return {
        "message": f"Server '{req.name}' added." + (" Connected." if connected else " Not reachable yet."),
        "connected": connected,
        "server": new_config,
    }


@app.patch("/servers/{server_name}")
async def edit_server(server_name: str, req: EditServerRequest, user: TokenUser = Depends(get_current_user)):
    """Edit an MCP server's description or URL."""
    if req.url:
        validate_server_url(req.url)

    if not user_db.update_user_server(user.username, server_name, description=req.description, url=req.url):
        raise HTTPException(status_code=404, detail=f"Server '{server_name}' not found.")

    # Update in registry
    config = next((c for c in registry.server_configs if c["name"] == server_name), None)
    url_changed = False
    if config:
        if req.description is not None:
            config["description"] = req.description
        if req.url is not None:
            url_changed = req.url != config["url"]
            config["url"] = req.url

    if url_changed:
        registry.agents = [a for a in registry.agents if a["name"] != server_name]
        if server_name not in registry.inactive_servers:
            registry.inactive_servers.append(server_name)
        await registry.refresh_mcp_servers()

    logger.info(f"[{user.username}] Edited server: {server_name}")
    return {"message": f"Server '{server_name}' updated."}


@app.delete("/servers/{server_name}")
async def remove_server(server_name: str, user: TokenUser = Depends(get_current_user)):
    """Remove an MCP server for the current user."""
    if not user_db.delete_user_server(user.username, server_name):
        raise HTTPException(status_code=404, detail=f"Server '{server_name}' not found.")

    # Remove from registry
    registry.server_configs = [c for c in registry.server_configs if c["name"] != server_name]
    registry.agents = [a for a in registry.agents if a["name"] != server_name]
    if server_name in registry.inactive_servers:
        registry.inactive_servers.remove(server_name)

    logger.info(f"[{user.username}] Removed server: {server_name}")
    return {"message": f"Server '{server_name}' removed."}


@app.get("/servers/{server_name}/tools")
async def get_server_tools(server_name: str, user: TokenUser = Depends(get_current_user)):
    """List tools for a connected MCP server."""
    agent_spec = registry.get_agent(server_name)
    if not agent_spec:
        raise HTTPException(status_code=404, detail=f"Server '{server_name}' not connected.")

    agent = agent_spec["agent"]
    tools = []

    # Extract tools directly from the saved list in the agent_spec
    tools_list = agent_spec.get("tools", [])
    for tool in tools_list:
        tools.append({"name": getattr(tool, "name", ""), "description": getattr(tool, "description", "")})

    return {"server": server_name, "tools": tools, "count": len(tools)}


@app.post("/servers/refresh")
async def refresh_servers(user: TokenUser = Depends(get_current_user)):
    """Hot-reload MCP servers for the current user."""
    # Reload user's servers into the registry
    user_configs = user_db.get_user_servers(user.username)
    for config in user_configs:
        if config["name"] not in [c["name"] for c in registry.server_configs]:
            registry.server_configs.append(config)
            registry.inactive_servers.append(config["name"])

    logger.info(f"[{user.username}] Refreshing MCP servers...")
    result = await registry.refresh_mcp_servers()
    return {
        "message": "Server refresh complete",
        "connected_mcp_servers": result["connected"],
        "inactive_mcp_servers": result["inactive"],
    }


# =============================================================================
# ADMIN-ONLY ENDPOINTS
# =============================================================================

@app.get("/admin/status", dependencies=[Depends(require_admin)])
async def admin_status():
    """Full system diagnostics. (Admin only)"""
    import platform
    return {
        "system": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "llm_provider": os.getenv("LLM_PROVIDER", "ollama"),
            "llm_model": os.getenv("LLM_MODEL", "llama3.2"),
            "database": os.getenv("DATABASE_URL", "sqlite:///conversations.db").split("://")[0],
            "redis": "connected" if os.getenv("REDIS_URL") else "in-memory",
        },
        "agents": [a["name"] for a in registry.agents],
        "inactive_servers": registry.inactive_servers,
        "server_configs": [{"name": c["name"], "url": c["url"]} for c in registry.server_configs],
        "conversation_count": len(conversation_db.get_conversations(limit=9999)),
    }


@app.get("/admin/conversations", dependencies=[Depends(require_admin)])
async def admin_list_all_conversations(limit: int = 100):
    """List ALL conversations across all users. (Admin only)"""
    convos = conversation_db.get_conversations(user_id=None, limit=limit)
    return [
        {
            "id": c.id,
            "title": c.title,
            "created_at": str(c.created_at),
            "updated_at": str(c.updated_at),
        }
        for c in convos
    ]


@app.delete("/admin/conversations/clear", dependencies=[Depends(require_admin)])
async def admin_clear_all_conversations():
    """Delete ALL conversations. Destructive! (Admin only)"""
    convos = conversation_db.get_conversations(user_id=None, limit=9999)
    count = 0
    for c in convos:
        conversation_db.delete_conversation(c.id)
        count += 1
    logger.warning(f"[admin] Cleared all conversations: {count} deleted")
    return {"message": f"Deleted {count} conversations."}


@app.get("/admin/logs", dependencies=[Depends(require_admin)])
async def admin_get_logs(lines: int = 100, level: Optional[str] = None):
    """View recent application logs. (Admin only)"""
    log_file = os.path.join(os.path.dirname(__file__), "logs", "mosaic.log")

    if not os.path.exists(log_file):
        return {"logs": [], "message": "No log file found yet."}

    with open(log_file, "r") as f:
        all_lines = f.readlines()

    recent = all_lines[-lines:]
    if level:
        recent = [l for l in recent if f"| {level.upper()}" in l]

    return {"logs": [l.rstrip() for l in recent], "total_lines": len(all_lines)}


@app.get("/admin/logs/errors", dependencies=[Depends(require_admin)])
async def admin_get_error_logs(lines: int = 50):
    """View error-only logs. (Admin only)"""
    log_file = os.path.join(os.path.dirname(__file__), "logs", "mosaic_errors.log")

    if not os.path.exists(log_file):
        return {"logs": [], "message": "No errors logged yet."}

    with open(log_file, "r") as f:
        all_lines = f.readlines()

    recent = all_lines[-lines:]
    return {"logs": [l.rstrip() for l in recent], "total_lines": len(all_lines)}


@app.get("/admin/logs/requests", dependencies=[Depends(require_admin)])
async def admin_get_request_logs(lines: int = 100):
    """View HTTP request logs. (Admin only)"""
    log_file = os.path.join(os.path.dirname(__file__), "logs", "requests.log")

    if not os.path.exists(log_file):
        return {"logs": [], "message": "No request logs yet."}

    with open(log_file, "r") as f:
        all_lines = f.readlines()

    recent = all_lines[-lines:]
    return {"logs": [l.rstrip() for l in recent], "total_lines": len(all_lines)}


@app.get("/admin/config", dependencies=[Depends(require_admin)])
async def admin_get_config():
    """View current runtime configuration. (Admin only) Sensitive values are masked."""
    return {
        "llm_provider": os.getenv("LLM_PROVIDER", "ollama"),
        "llm_model": os.getenv("LLM_MODEL", "llama3.2"),
        "llm_base_url": os.getenv("LLM_BASE_URL", "default"),
        "database": os.getenv("DATABASE_URL", "sqlite:///conversations.db").split("@")[-1] if "@" in os.getenv("DATABASE_URL", "") else os.getenv("DATABASE_URL", "sqlite:///conversations.db"),
        "redis": "connected" if os.getenv("REDIS_URL") else "in-memory",
        "environment": os.getenv("ENVIRONMENT", "development"),
        "token_expire_hours": int(os.getenv("TOKEN_EXPIRE_HOURS", "24")),
        "login_rate_limit": int(os.getenv("LOGIN_RATE_LIMIT", "5")),
        "login_rate_window_sec": int(os.getenv("LOGIN_RATE_WINDOW", "300")),
        "allowed_origins": ALLOWED_ORIGINS,
        "log_level": os.getenv("LOG_LEVEL", "INFO"),
        "max_history_messages": int(os.getenv("MAX_HISTORY_MESSAGES", "10")),
        "tavily_key_set": bool(os.getenv("TAVILY_API_KEY")),
        "jwt_secret_set": bool(os.getenv("JWT_SECRET")),
    }


# =============================================================================
# JOBS — Background Task Queue (authenticated)
# =============================================================================

class JobSubmitRequest(BaseModel):
    job_type: str = Field(..., description="Job type: mcp_tool_call, rag_process, web_scrape, batch_chat")
    payload: dict = Field(default_factory=dict)
    priority: int = Field(default=5, ge=1, le=10000)
    timeout_seconds: int = Field(default=300, ge=1, le=86400)
    max_retries: int = Field(default=2, ge=0, le=10)


@app.post("/jobs")
async def submit_job(req: JobSubmitRequest, user: TokenUser = Depends(get_current_user)):
    """Submit a background job to the task queue."""
    try:
        from flowq.src.mosaic_bridge import submit_background_job, JOB_TYPES
    except ImportError:
        raise HTTPException(status_code=503, detail="FlowQ not available.")

    if req.job_type not in JOB_TYPES:
        raise HTTPException(status_code=400, detail=f"Unknown job type. Available: {list(JOB_TYPES.keys())}")

    job_id = await submit_background_job(
        job_type=req.job_type,
        payload=req.payload,
        user_id=user.username,
        priority=req.priority,
        timeout_seconds=req.timeout_seconds,
        max_retries=req.max_retries,
    )

    logger.info(f"[{user.username}] Submitted job: type={req.job_type} id={job_id}")
    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str, user: TokenUser = Depends(get_current_user)):
    """Get the status and result of a background job."""
    try:
        from flowq.src.mosaic_bridge import get_job_result
    except ImportError:
        raise HTTPException(status_code=503, detail="FlowQ not available.")

    result = await get_job_result(job_id)
    if result["status"] == "not_found":
        raise HTTPException(status_code=404, detail="Job not found.")
    return result


@app.delete("/jobs/{job_id}")
async def cancel_job_endpoint(job_id: str, user: TokenUser = Depends(get_current_user)):
    """Cancel a pending or queued job."""
    try:
        from flowq.src.mosaic_bridge import cancel_job
    except ImportError:
        raise HTTPException(status_code=503, detail="FlowQ not available.")

    success = await cancel_job(job_id)
    if not success:
        raise HTTPException(status_code=400, detail="Job could not be cancelled (may already be running).")
    return {"message": f"Job {job_id} cancelled."}
