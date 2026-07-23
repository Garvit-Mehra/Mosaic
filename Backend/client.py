#!/usr/bin/env python3
"""
Mosaic - Modular Multi-Agent Client

Stateless architecture: each request loads context from the database,
processes through the agent, and returns. No in-memory conversation state.
Supports multiple concurrent workers safely.
"""

import os
import logging
import aiohttp
import datetime
from typing import List, Dict, Any, Optional, Tuple, AsyncGenerator

from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_mcp_adapters.client import MultiServerMCPClient

from utils.RAGTools import load_document, query_documents, list_documents, clear_documents
from utils.llm import get_classifier_model, get_agent_model
from utils.ConversationDB import ConversationManager

load_dotenv()

from utils.logger import setup_logging, get_logger
setup_logging(level=os.getenv("LOG_LEVEL", "INFO"))
logger = get_logger("client")

logging.getLogger("httpx").setLevel(logging.WARNING)

# Configuration
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "10"))

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY not set in environment variables.")


# =============================================================================
# MCP Server Utilities
# =============================================================================

class MCPClientManager:
    def __init__(self, server_configs):
        self.server_dict = {}
        for config in server_configs:
            entry = {k: v for k, v in config.items() if k in ("url", "transport", "command", "args")}
            if "transport" not in entry:
                entry["transport"] = "sse"
            self.server_dict[config["name"]] = entry
        self.client = MultiServerMCPClient(self.server_dict)

    def get_client(self):
        return self.client


async def is_server_active(url: str) -> bool:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=2)) as resp:
                return resp.status == 200
    except Exception:
        return False


async def get_mcp_tools(client_manager: MCPClientManager, server_name: str):
    client = client_manager.get_client()
    if not client:
        return []
    try:
        tools = await client.get_tools(server_name=server_name)
        logger.info(f"Loaded {len(tools)} tools for {server_name}")
        return tools
    except Exception as e:
        logger.error(f"Failed to load tools for {server_name}: {e}")
        return []


# =============================================================================
# Agent Registry (initialized once, shared across requests)
# =============================================================================

class AgentRegistry:
    """
    Manages available agents. Initialized once at startup.
    Agents are stateless — they don't hold conversation state.
    Conversation context is passed in per-request.
    """

    def __init__(self):
        self.agents: List[Dict[str, Any]] = []
        self.inactive_servers: List[str] = []
        self.server_configs: List[Dict[str, Any]] = []
        self._initialized = False

    async def initialize(self, server_configs: List[Dict[str, Any]], web_search: bool = True):
        """Build all agents. Called once at startup."""
        if self._initialized:
            return

        self.server_configs = server_configs
        agents = []

        # General agent
        agents.append({
            "name": "general",
            "description": "For writing, coding, math, explanations, creative tasks, opinions, and all general questions.",
            "agent": create_react_agent(
                get_agent_model(),
                tools=[],
                prompt=(
                    "You are Mosaic, a helpful AI assistant. "
                    "You have NO tools. Never output JSON or function calls. "
                    "Always respond directly in plain text or markdown. "
                    "Be concise and helpful. Write code when asked. "
                    "Do not restate the question or add unnecessary preamble."
                ),
                checkpointer=MemorySaver()
            ),
        })

        # Web search agent
        if web_search and TAVILY_API_KEY:
            agents.append({
                "name": "web",
                "description": "ONLY for live/real-time info: current news, weather, stock prices, sports scores.",
                "agent": create_react_agent(
                    get_agent_model(),
                    tools=[TavilySearch(api_key=TAVILY_API_KEY, max_results=2)],
                    prompt=(
                        "You are Mosaic's web search agent. "
                        "Use the search tool ONLY when the user needs live/current information. "
                        "After searching, summarize the results in plain text. "
                        "Never output raw JSON. Be brief and clear."
                    ),
                    checkpointer=MemorySaver()
                ),
            })

        # MCP server agents
        if server_configs:
            active_configs = []
            for config in server_configs:
                if await is_server_active(config["url"]):
                    active_configs.append(config)
                else:
                    logger.info(f"MCP server '{config['name']}' not running — skipped")
                    self.inactive_servers.append(config["name"])

            if active_configs:
                client_manager = MCPClientManager(active_configs)
                for config in active_configs:
                    try:
                        mcp_tools = await get_mcp_tools(client_manager, config["name"])
                        agents.append({
                            "name": config["name"],
                            "description": config["description"],
                            "agent": create_react_agent(
                                get_agent_model(),
                                tools=mcp_tools,
                                prompt=(
                                    f"You are Mosaic's {config['name'].replace('_', ' ').title()} agent. "
                                    f"{config['description']} "
                                    "Give only the requested information."
                                ),
                                checkpointer=MemorySaver()
                            ),
                        })
                    except Exception as e:
                        logger.warning(f"Failed to load '{config['name']}': {e}")
                        self.inactive_servers.append(config["name"])

        # RAG agent
        agents.append({
            "name": "rag",
            "description": "ONLY when user explicitly asks about a loaded PDF, document, or file they uploaded.",
            "agent": create_react_agent(
                get_agent_model(),
                tools=[load_document, query_documents, list_documents, clear_documents],
                prompt=(
                    "You are Mosaic's RAG agent. "
                    "Answer strictly from the loaded documents. "
                    "If the answer is not in the docs, say so briefly."
                ),
                checkpointer=MemorySaver()
            ),
        })

        self.agents = agents
        self._initialized = True
        logger.info(f"Initialized agents: {[a['name'] for a in agents]}. Inactive: {self.inactive_servers}")

    async def refresh_mcp_servers(self) -> Dict[str, Any]:
        """Hot-reload: detect newly started/stopped MCP servers."""
        connected = []
        active_names = {a["name"] for a in self.agents if a["name"] not in ("general", "web", "rag")}

        servers_to_check = [
            c for c in self.server_configs
            if c["name"] in self.inactive_servers or c["name"] not in active_names
        ]

        if servers_to_check:
            client_manager = MCPClientManager(servers_to_check)
            for config in servers_to_check:
                if await is_server_active(config["url"]):
                    try:
                        mcp_tools = await get_mcp_tools(client_manager, config["name"])
                        self.agents.append({
                            "name": config["name"],
                            "description": config["description"],
                            "agent": create_react_agent(
                                get_agent_model(),
                                tools=mcp_tools,
                                prompt=(
                                    f"You are Mosaic's {config['name'].replace('_', ' ').title()} agent. "
                                    f"{config['description']} "
                                    "Give only the requested information."
                                ),
                                checkpointer=MemorySaver()
                            ),
                        })
                        if config["name"] in self.inactive_servers:
                            self.inactive_servers.remove(config["name"])
                        connected.append(config["name"])
                    except Exception as e:
                        logger.error(f"Failed to hot-load {config['name']}: {e}")

        # Check for gone-offline servers
        for agent in list(self.agents):
            if agent["name"] in ("general", "web", "rag"):
                continue
            config = next((c for c in self.server_configs if c["name"] == agent["name"]), None)
            if config and not await is_server_active(config["url"]):
                self.agents.remove(agent)
                if agent["name"] not in self.inactive_servers:
                    self.inactive_servers.append(agent["name"])

        all_active = [a["name"] for a in self.agents if a["name"] not in ("general", "web", "rag")]
        return {"connected": all_active, "inactive": self.inactive_servers}

    def get_agent(self, name: str) -> Optional[Dict[str, Any]]:
        return next((a for a in self.agents if a["name"] == name), None)


# =============================================================================
# Stateless Request Handler
# =============================================================================

class MosaicHandler:
    """
    Stateless request handler. Each call:
    1. Loads conversation history from DB
    2. Classifies the query → picks an agent
    3. Invokes the agent with full context
    4. Returns the response
    
    No in-memory state between requests. Safe for multiple workers.
    """

    def __init__(self, registry: AgentRegistry, conversation_db: ConversationManager):
        self.registry = registry
        self.db = conversation_db
        self.classifier = get_classifier_model()

    def _build_context(self, conversation_id: Optional[int], user_message: str) -> List[Dict[str, str]]:
        """Load conversation history from DB and build the message list."""
        messages = []

        # System context
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        messages.append({
            "role": "system",
            "content": (
                f"Today's date is {today}. "
                "Do not mention the date unless asked. "
                "Be concise and direct."
            )
        })

        # Load history from DB
        if conversation_id:
            prior = self.db.get_conversation_context(conversation_id, max_messages=MAX_HISTORY_MESSAGES)
            messages.extend(prior)

        # Add current user message
        messages.append({"role": "user", "content": user_message})
        return messages

    async def _classify(self, user_input: str, history: List[Dict[str, str]]) -> str:
        """Route the query to the correct agent."""
        prompt = "You are a router. Pick ONE agent name for the user's query.\n\n"
        prompt += "Rules:\n"
        prompt += "1. Use 'web' ONLY for live/current info (news, weather, scores).\n"
        prompt += "2. Use 'rag' ONLY when user mentions a loaded document/PDF/file.\n"
        prompt += "3. Use 'general' for ALL other queries.\n"
        prompt += "4. When in doubt, use 'general'.\n\n"
        prompt += "Agents:\n"
        for agent in self.registry.agents:
            prompt += f"- {agent['name']}: {agent['description']}\n"
        prompt += f"\nUser query: {user_input}\n\n"
        prompt += "Reply with ONLY the agent name."

        result = await self.classifier.ainvoke(prompt)
        name = result.content.strip().split()[0].lower()

        # Validate
        if not self.registry.get_agent(name):
            if name in self.registry.inactive_servers:
                return "__inactive__:" + name
            return "general"
        return name

    async def chat(self, message: str, conversation_id: Optional[int] = None, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Process a chat message (non-streaming). Stateless."""
        messages = self._build_context(conversation_id, message)

        agent_name = await self._classify(message, messages)

        if agent_name.startswith("__inactive__:"):
            name = agent_name.split(":")[1]
            return {"response": f"Sorry, the {name.replace('_', ' ')} agent is currently unavailable.", "agent": "error"}

        agent_spec = self.registry.get_agent(agent_name)
        if not agent_spec:
            agent_spec = self.registry.get_agent("general")

        try:
            config = {"configurable": {"thread_id": f"{user_id or 'anon'}_{conversation_id or 'new'}"}}
            result = await agent_spec["agent"].ainvoke({"messages": messages}, config)
            ai_message = result["messages"][-1].content
            return {"response": ai_message, "agent": agent_spec["name"]}
        except Exception as e:
            logger.error(f"Agent error: {e}")
            return {"response": "I encountered an error while processing your request.", "agent": "error"}

    async def chat_stream(self, message: str, conversation_id: Optional[int] = None, user_id: Optional[str] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """Process a chat message with streaming. Stateless."""
        messages = self._build_context(conversation_id, message)

        agent_name = await self._classify(message, messages)

        if agent_name.startswith("__inactive__:"):
            name = agent_name.split(":")[1]
            yield {"type": "error", "content": f"Sorry, the {name.replace('_', ' ')} agent is currently unavailable."}
            return

        agent_spec = self.registry.get_agent(agent_name)
        if not agent_spec:
            agent_spec = self.registry.get_agent("general")

        yield {"type": "agent", "agent": agent_spec["name"]}

        try:
            config = {"configurable": {"thread_id": f"{user_id or 'anon'}_{conversation_id or 'new'}"}}
            full_response = ""

            async for event in agent_spec["agent"].astream_events(
                {"messages": messages}, config, version="v2"
            ):
                if event.get("event") == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        full_response += chunk.content
                        yield {"type": "token", "content": chunk.content}

            yield {"type": "done", "full_response": full_response}

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield {"type": "error", "content": "I encountered an error while processing your request."}
