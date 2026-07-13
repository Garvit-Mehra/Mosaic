from fastapi import FastAPI, HTTPException
from client import MosaicAPI
from pydantic import BaseModel
from typing import Optional, List
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from utils.ConversationDB import ConversationManager
import json

# --- Server configuration (all MCP servers are optional) ---
SERVER_CONFIGS = [
    {
        "name": "database_server",
        "description": "Handles database operations — create, read, update, delete tables and data.",
        "url": "http://localhost:8000/sse"
    },
    {
        "name": "calendar_server",
        "description": "Manages Google Calendar — view, create, edit, delete events and check availability.",
        "url": "http://localhost:8010/sse"
    },
]

mosaic_api = MosaicAPI(SERVER_CONFIGS, web_search=True, model_config="llama3.2")
conversation_db = ConversationManager()


# --- Lifespan (runs on startup/shutdown) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    await mosaic_api.initialize()
    yield


app = FastAPI(lifespan=lifespan)

# Allow all origins for dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request/Response Models ---
class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[int] = None
    user_id: Optional[str] = None


class ConversationCreateRequest(BaseModel):
    title: str
    user_id: Optional[str] = None


class ConversationUpdateRequest(BaseModel):
    title: str


# --- Chat Endpoint ---
@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Send a message and get a response.
    If conversation_id is provided, messages are persisted to that conversation.
    If not, a new conversation is auto-created from the first message.
    """
    conv_id = req.conversation_id

    # Auto-create conversation if none provided
    if conv_id is None:
        title = req.message[:50] + ("..." if len(req.message) > 50 else "")
        convo = conversation_db.create_conversation(title=title, user_id=req.user_id)
        conv_id = convo.id

    # Get AI response
    result = await mosaic_api.chat(
        req.message,
        conversation_id=conv_id,
        user_id=req.user_id,
    )

    # Persist both messages
    conversation_db.add_message(conv_id, "user", req.message)
    conversation_db.add_message(
        conv_id, "assistant", result["response"], agent=result.get("agent")
    )

    return {
        "response": result["response"],
        "agent": result.get("agent"),
        "conversation_id": conv_id,
    }


# --- Streaming Chat Endpoint ---
@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    Stream a response via Server-Sent Events.
    Each event is a JSON object with a 'type' field:
      - {"type": "agent", "agent": "general"}
      - {"type": "token", "content": "Hello"}
      - {"type": "done", "full_response": "...", "conversation_id": 5}
      - {"type": "error", "content": "..."}
    """
    conv_id = req.conversation_id

    # Auto-create conversation if none provided
    if conv_id is None:
        title = req.message[:50] + ("..." if len(req.message) > 50 else "")
        convo = conversation_db.create_conversation(title=title, user_id=req.user_id)
        conv_id = convo.id

    async def event_generator():
        full_response = ""
        agent_name = None

        async for chunk in mosaic_api.chat_stream(
            req.message,
            conversation_id=conv_id,
            user_id=req.user_id,
        ):
            if chunk["type"] == "agent":
                agent_name = chunk["agent"]
            elif chunk["type"] == "done":
                full_response = chunk.get("full_response", "")
            
            yield f"data: {json.dumps(chunk)}\n\n"

        # Persist messages after streaming is done
        conversation_db.add_message(conv_id, "user", req.message)
        if full_response:
            conversation_db.add_message(conv_id, "assistant", full_response, agent=agent_name)

        # Send final event with conversation_id
        yield f"data: {json.dumps({'type': 'done', 'conversation_id': conv_id, 'full_response': full_response, 'agent': agent_name})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# --- Conversation CRUD ---
@app.post("/conversations")
async def create_conversation(req: ConversationCreateRequest):
    """Create a new empty conversation."""
    convo = conversation_db.create_conversation(title=req.title, user_id=req.user_id)
    return {"id": convo.id, "title": convo.title, "created_at": str(convo.created_at)}


@app.get("/conversations")
async def list_conversations(user_id: Optional[str] = None, limit: int = 50):
    """List conversations, optionally filtered by user_id."""
    convos = conversation_db.get_conversations(user_id=user_id, limit=limit)
    return [
        {
            "id": c.id,
            "title": c.title,
            "created_at": str(c.created_at),
            "updated_at": str(c.updated_at),
        }
        for c in convos
    ]


@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: int):
    """Get all messages for a conversation."""
    convo = conversation_db.get_conversation(conversation_id)
    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found")

    messages = conversation_db.get_messages(conversation_id)
    return {
        "id": convo.id,
        "title": convo.title,
        "messages": [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "agent": m.agent,
                "timestamp": str(m.timestamp),
            }
            for m in messages
        ],
    }


@app.patch("/conversations/{conversation_id}")
async def update_conversation(conversation_id: int, req: ConversationUpdateRequest):
    """Update a conversation's title."""
    success = conversation_db.update_conversation_title(conversation_id, req.title)
    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"message": "Conversation updated"}


@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: int):
    """Delete a conversation and all its messages."""
    success = conversation_db.delete_conversation(conversation_id)
    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"message": "Conversation deleted"}


@app.get("/conversations/{conversation_id}/stats")
async def get_conversation_stats(conversation_id: int):
    """Get statistics for a conversation."""
    convo = conversation_db.get_conversation(conversation_id)
    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation_db.get_conversation_stats(conversation_id)


# --- Server Management ---
class AddServerRequest(BaseModel):
    name: str
    description: str
    url: str


@app.get("/servers")
async def list_servers():
    """List all configured MCP servers with their status."""
    from client import is_server_active
    servers = []
    for config in mosaic_api.server_configs:
        active = await is_server_active(config["url"])
        # Check if agent is loaded
        agent_loaded = False
        if mosaic_api.mosaic:
            agent_loaded = any(s["name"] == config["name"] for s in mosaic_api.mosaic.agent_specs)
        servers.append({
            "name": config["name"],
            "description": config["description"],
            "url": config["url"],
            "active": active,
            "agent_loaded": agent_loaded,
        })
    return {"servers": servers}


@app.post("/servers")
async def add_server(req: AddServerRequest):
    """
    Add a new MCP server configuration and immediately try to connect.
    If the server is reachable, its tools are loaded and an agent is created.
    """
    # Check if server with same name already exists
    existing_names = [c["name"] for c in mosaic_api.server_configs]
    if req.name in existing_names:
        raise HTTPException(status_code=400, detail=f"Server '{req.name}' already exists.")

    new_config = {
        "name": req.name,
        "description": req.description,
        "url": req.url,
    }

    # Add to the config list
    mosaic_api.server_configs.append(new_config)
    if mosaic_api.mosaic:
        mosaic_api.mosaic.server_configs.append(new_config)
        mosaic_api.mosaic.inactive_agents.append(req.name)

    # Try to connect immediately
    result = await mosaic_api.refresh_servers()

    connected = req.name in result.get("connected", [])
    return {
        "message": f"Server '{req.name}' added." + (" Connected and agent loaded." if connected else " Server not reachable — will retry on next refresh."),
        "connected": connected,
        "server": new_config,
    }


@app.delete("/servers/{server_name}")
async def remove_server(server_name: str):
    """Remove an MCP server configuration and its agent."""
    config = next((c for c in mosaic_api.server_configs if c["name"] == server_name), None)
    if not config:
        raise HTTPException(status_code=404, detail=f"Server '{server_name}' not found.")

    # Remove from config
    mosaic_api.server_configs.remove(config)

    # Remove agent if loaded
    if mosaic_api.mosaic:
        mosaic_api.mosaic.server_configs = [c for c in mosaic_api.mosaic.server_configs if c["name"] != server_name]
        mosaic_api.mosaic.agent_specs = [s for s in mosaic_api.mosaic.agent_specs if s["name"] != server_name]
        if server_name in mosaic_api.mosaic.inactive_agents:
            mosaic_api.mosaic.inactive_agents.remove(server_name)

    return {"message": f"Server '{server_name}' removed."}


@app.get("/servers/{server_name}/tools")
async def get_server_tools(server_name: str):
    """Get the list of tools available on a connected MCP server."""
    if not mosaic_api.mosaic:
        raise HTTPException(status_code=503, detail="System not initialized yet.")

    agent_spec = next((s for s in mosaic_api.mosaic.agent_specs if s["name"] == server_name), None)
    if not agent_spec:
        raise HTTPException(status_code=404, detail=f"Server '{server_name}' is not connected or not found.")

    # Extract tool info from the agent
    agent = agent_spec["agent"]
    tools = []
    # LangGraph react agents store tools in the agent's tool node
    if hasattr(agent, 'nodes') and 'tools' in agent.nodes:
        tool_node = agent.nodes['tools']
        if hasattr(tool_node, 'tools_by_name'):
            for name, tool in tool_node.tools_by_name.items():
                tools.append({
                    "name": name,
                    "description": getattr(tool, 'description', ''),
                })

    return {"server": server_name, "tools": tools}


@app.get("/status")
async def get_status():
    """Get current system status — active agents and inactive MCP servers."""
    return await mosaic_api.get_status()


@app.post("/servers/refresh")
async def refresh_servers():
    """
    Hot-reload MCP servers. Call this after starting a database or calendar server
    to have Mosaic pick it up without restarting the backend.
    """
    result = await mosaic_api.refresh_servers()
    return {
        "message": "Server refresh complete",
        "connected_mcp_servers": result["connected"],
        "inactive_mcp_servers": result["inactive"],
    }
