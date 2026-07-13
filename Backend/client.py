#!/usr/bin/env python3
"""
Mosaic - Modular Multi-Agent Tools for Python

A modern toolkit for building, combining, and experimenting with modular multi-agent tools.

Authors: Mosiac Team
Version: 1.3.1
License: Non-Commercial, No-Distribution (Based on MIT)
"""

import os
import json
import asyncio
import logging
import aiohttp
import datetime
from collections import deque
from typing import List, Dict, Any, Optional, Tuple, AsyncGenerator

# Third-party imports
from dotenv import load_dotenv
from langchain_ollama.chat_models import ChatOllama
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_mcp_adapters.client import MultiServerMCPClient

# Local imports
from utils.RAGTools import load_document, query_documents, list_documents, clear_documents

# Load environment variables from .env file
load_dotenv()

# Configure logging
from utils.logger import setup_logging, get_logger
setup_logging(level=os.getenv("LOG_LEVEL", "INFO"))
logger = get_logger("client")

# Suppress verbose HTTP logs from `httpx`
logging.getLogger("httpx").setLevel(logging.WARNING)

# Core configuration constants
MAX_HISTORY_EXCHANGES = 5

# Validate required API key early
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY not set in environment variables. Please check your .env file.")

# --- MCPClientManager for Modular Clients ---
class MCPClientManager:
    def __init__(self, server_configs):
        # Build a dict of server configs for MultiServerMCPClient
        self.server_dict = {}
        for config in server_configs:
            # Keep only relevant keys
            entry = {k: v for k, v in config.items() if k in ("url", "transport", "command", "args")}
            # Default transport is SSE
            if "transport" not in entry:
                entry["transport"] = "sse"
            self.server_dict[config["name"]] = entry
        # Create one unified client for all servers
        self.client = MultiServerMCPClient(self.server_dict)

    def get_client(self):
        # Return the underlying MultiServerMCPClient
        return self.client

# Async health check for MCP server
async def is_server_active(url: str) -> bool:
    try:
        # Attempt a GET request with a short timeout
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=2) as resp:
                return resp.status == 200
    except Exception:
        return False

# --- Load MCP tools for a specific server ---
async def get_mcp_tools_modular(client_manager, server_name: str):
    client = client_manager.get_client()
    if not client:
        logger.error(f"No MCP client found for server: {server_name}")
        return []
    try:
        logger.info(f"Attempting to load MCP tools for {server_name}")
        # Use `get_tools` call to retrieve server tools
        tools = await client.get_tools(server_name=server_name)
        logger.info(f"Successfully loaded {len(tools)} tools for {server_name}")
        return tools
    except Exception as e:
        logger.error(f"Failed to load MCP tools for {server_name}: {e}")
        logger.error(f"Error type: {type(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return []

class Mosaic:
    """
    Main Mosaic client class that manages the multi-agent system.
    MCP servers are optional — the system works without them and can
    hot-reload new servers at any time via refresh_mcp_servers().
    """
    def __init__(self, agent_specs, inactive_agents, server_configs, web_search, conversation_id=None, user_id=None, model_config=None):
        # Conversation history (system + user + assistant messages)
        self.conversation_id = conversation_id if conversation_id is not None else -1
        self.user_id = user_id
        self.history = deque(maxlen=MAX_HISTORY_EXCHANGES * 2)
        self.last_agent_used = None
        self.classifier_llm = ChatOllama(model=model_config, temperature=0.0, reasoning=False)  # Used for agent selection
        self.agent_specs = agent_specs
        self.inactive_agents = inactive_agents
        self.server_configs = server_configs
        self.model_config = model_config
        self.web_search = web_search
        # Add system date context for new conversation
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        self.history.append({
            "role": "system",
            "content": (
                f"For your context only: today's date is {today}. "
                "Do not mention or repeat the date unless the user explicitly asks. "
                "All answers must be short and direct."
            )
        })

    async def refresh_mcp_servers(self) -> Dict[str, Any]:
        """
        Hot-reload MCP servers. Checks which configured servers are now active,
        creates agents for newly available ones, and marks disappeared ones as inactive.
        Can be called at any time while the system is running.
        
        Returns a dict with 'connected' and 'inactive' server lists.
        """
        connected = []
        still_inactive = []

        # Build a set of server names that already have active agents
        active_mcp_names = {
            spec["name"] for spec in self.agent_specs
            if spec["name"] not in ("general", "web", "rag")
        }

        # Only attempt servers that are currently inactive
        servers_to_check = [
            config for config in self.server_configs
            if config["name"] in self.inactive_agents or config["name"] not in active_mcp_names
        ]

        if not servers_to_check:
            return {"connected": list(active_mcp_names), "inactive": self.inactive_agents}

        # Check each inactive server
        client_manager = MCPClientManager(servers_to_check)
        for config in servers_to_check:
            if await is_server_active(config["url"]):
                try:
                    mcp_tools = await get_mcp_tools_modular(client_manager, config["name"])
                    new_agent = {
                        "name": config["name"],
                        "description": config["description"],
                        "agent": create_react_agent(
                            ChatOllama(model=self.model_config, reasoning=False),
                            tools=mcp_tools,
                            prompt=(
                                f"You are Mosaic's {config['name'].replace('_', ' ').title()} agent. "
                                f"{config['description']} "
                                "Give only the requested information. "
                                "No extra commentary, no repeating context, no formatting unless required."
                            ),
                            checkpointer=MemorySaver()
                        ),
                        "thread_id": f"thread_{config['name']}"
                    }
                    # Add to active agents if not already there
                    if config["name"] not in active_mcp_names:
                        self.agent_specs.append(new_agent)
                    # Remove from inactive list
                    if config["name"] in self.inactive_agents:
                        self.inactive_agents.remove(config["name"])
                    connected.append(config["name"])
                    logger.info(f"Hot-loaded MCP server: {config['name']}")
                except Exception as e:
                    logger.error(f"Failed to hot-load {config['name']}: {e}")
                    still_inactive.append(config["name"])
            else:
                still_inactive.append(config["name"])

        # Also check if any currently active MCP agents have gone offline
        for spec in list(self.agent_specs):
            if spec["name"] in ("general", "web", "rag"):
                continue
            config = next((c for c in self.server_configs if c["name"] == spec["name"]), None)
            if config and not await is_server_active(config["url"]):
                self.agent_specs.remove(spec)
                if spec["name"] not in self.inactive_agents:
                    self.inactive_agents.append(spec["name"])
                logger.warning(f"MCP server went offline: {spec['name']}")

        all_active_mcp = [s["name"] for s in self.agent_specs if s["name"] not in ("general", "web", "rag")]
        return {"connected": all_active_mcp, "inactive": self.inactive_agents}

    @classmethod
    def create(cls, server_configs: List[Dict[str, Any]], web_search: bool = True, model_config=None):
        # Async wrapper to allow synchronous entrypoint
        async def _acreate():
            # Create a temporary instance to initialize agents
            temp_instance = cls([], [], server_configs, web_search, model_config=model_config)
            agent_specs, inactive_agents = await temp_instance._initialize_agents(server_configs, web_search)
            return cls(agent_specs, inactive_agents, server_configs, web_search, model_config=model_config)
        return asyncio.run(_acreate())

    def run(self):
        # Main loop (blocking) for CLI mode
        asyncio.run(self._arun())

    async def _arun(self):
        # CLI interface with live agent routing and streaming
        print("Mosaic - Modular Multi-Agent Tools for Python")
        print("=" * 50)
        print("Type 'exit' or 'quit' to end the session")
        print("Available agents:", ", ".join([spec["name"].replace("_", " ").title() for spec in self.agent_specs]))
        print("=" * 50)
        agent_display_names = {spec["name"]: spec["name"].replace("_", "-").title() + "-agent" for spec in self.agent_specs}
        while True:
            try:
                user_input = input("\nYou: ").strip()
                if user_input.lower() in {"exit", "quit"}:
                    print("\nGoodbye! Thank you for using Mosaic.")
                    break
                if not user_input:
                    continue
                # Process message and route to correct agent
                response, agent_name, conv_id = await self.process_message(user_input)
                display_name = agent_display_names.get(agent_name, agent_name.title())
                print(f"Mosaic ({display_name}): {response}")
            except KeyboardInterrupt:
                print("\n\nSession interrupted. Goodbye!")
                break
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                print(f"\nAn unexpected error occurred: {e}")

    async def _initialize_agents(self, server_configs, web_search) -> Tuple[List[Dict[str, Any]], List[str]]:
        agents = []
        inactive = []

        # Always-available general agent
        agents.append({
            "name": "general",
            "description": "For writing, coding, math, explanations, creative tasks, opinions, and all general questions.",
            "agent": create_react_agent(
                ChatOllama(model=self.model_config, reasoning=False),
                tools=[],
                prompt=(
                    "You are Mosaic's general assistant. "
                    "Respond directly and concisely. "
                    "Do not restate instructions, system info, or the current date unless explicitly asked. "
                    "Avoid lists, explanations, or formatting unless the user requests it."
                ),
                checkpointer=MemorySaver()
            ),
            "thread_id": "thread_general"
        })

        # Optional web search agent
        if web_search:
            agents.append({
                "name": "web",
                "description": "ONLY for live/real-time info: current news, weather, stock prices, sports scores.",
                "agent": create_react_agent(
                    ChatOllama(model=self.model_config, reasoning=False),
                    tools=[TavilySearch(api_key=TAVILY_API_KEY, max_results=2)],
                    prompt=(
                        "You are Mosaic's web agent. "
                        "Use search only for real-time data. "
                        "Answer with the result directly, no metadata, no citations, no reasoning. "
                        "Be brief and clear. "
                        "Never explain how you searched unless asked."
                    ),
                    checkpointer=MemorySaver()
                ),
                "thread_id": "thread_web"
            })

        # MCP agents from server configs (all optional — skip silently if unavailable)
        if server_configs:
            active_configs = []
            for config in server_configs:
                if await is_server_active(config["url"]):
                    active_configs.append(config)
                else:
                    logger.info(f"MCP server '{config['name']}' not running — skipped (can be hot-loaded later)")
                    inactive.append(config["name"])

            if active_configs:
                client_manager = MCPClientManager(active_configs)
                for config in active_configs:
                    try:
                        mcp_tools = await get_mcp_tools_modular(client_manager, config["name"])
                        agents.append({
                            "name": config["name"],
                            "description": config["description"],
                            "agent": create_react_agent(
                                ChatOllama(model=self.model_config, reasoning=False),
                                tools=mcp_tools,
                                prompt=(
                                    f"You are Mosaic's {config['name'].replace('_', ' ').title()} agent. "
                                    f"{config['description']} "
                                    "Give only the requested information. "
                                    "No extra commentary, no repeating context, no formatting unless required."
                                ),
                                checkpointer=MemorySaver()
                            ),
                            "thread_id": f"thread_{config['name']}"
                        })
                    except Exception as e:
                        logger.warning(f"Failed to load tools for '{config['name']}': {e}. Marking inactive.")
                        inactive.append(config["name"])

        # RAG agent for document queries
        agents.append({
            "name": "rag",
            "description": "ONLY when user explicitly asks about a loaded PDF, document, or file they uploaded.",
            "agent": create_react_agent(
                ChatOllama(model=self.model_config, reasoning=False),
                tools=[load_document, query_documents, list_documents, clear_documents],
                prompt=(
                    "You are Mosaic's RAG agent. "
                    "Answer strictly from the loaded documents. "
                    "If the answer is not in the docs, say so briefly. "
                    "Do not add extra details or use formatting unless the user asks."
                ),
                checkpointer=MemorySaver()
            ),
            "thread_id": "thread_rag"
        })

        logger.info(f"Initialized agents: {[a['name'] for a in agents]}. Inactive MCP servers: {inactive}")
        return agents, inactive

    def _build_classification_prompt(self, user_input: str, conversation_context: deque, last_agent: Optional[str]) -> str:
        """
        Build a strict classification prompt for routing queries to agents.
        Returns a string with clear rules and no room for extra text in output.
        """

        # Core instructions
        prompt = (
            "You are a router. Pick ONE agent name for the user's query.\n\n"
            "Rules:\n"
            "1. If the query is a follow-up (e.g., 'it', 'that', 'more'), use the last agent.\n"
        )

        # Add dynamic rules based on available agents
        if any(spec['name'] == 'web' for spec in self.agent_specs):
            prompt += "2. Use 'web' ONLY for questions needing live/current info (news, weather, scores, stock prices).\n"
        prompt += "3. Use 'rag' ONLY when the user explicitly mentions a loaded document, PDF, or file.\n"
        prompt += "4. Use 'general' for ALL other queries: writing, coding, math, explanations, creative tasks, opinions, summaries.\n"
        prompt += "5. When in doubt, use 'general'.\n\n"

        # Append available agents
        prompt += "Agents:\n"
        for spec in self.agent_specs:
            prompt += f"- {spec['name']}: {spec['description']}\n"

        # Last agent reference
        if last_agent:
            prompt += f"\nLast agent used: {last_agent}\n"

        # Final query
        prompt += f"\nUser query: {user_input}\n\n"

        # Hard stop for output format
        prompt += "Reply with ONLY the agent name. Nothing else."

        return prompt

    async def _prepare_message(self, user_input: str) -> Tuple[str, Dict[str, Any], int]:
        # Common logic for processing user input and selecting agent
        # Append user message to history
        self.history.append({"role": "user", "content": user_input})
        # Build classification prompt
        prompt = self._build_classification_prompt(user_input, self.history, self.last_agent_used)
        # Get classification from LLM
        classification = await self.classifier_llm.ainvoke(prompt)
        agent_name = classification.content.strip().split()[0].lower()
        # Lookup matching agent spec
        agent_spec = next((spec for spec in self.agent_specs if spec["name"] == agent_name), None)
        if agent_spec is None:
            if hasattr(self, "inactive_agents") and agent_name in self.inactive_agents:
                msg = f"Sorry, the {agent_name.replace('_', ' ')} agent is currently unavailable."
                self.history.append({"role": "assistant", "content": msg})
                return msg, {"name": agent_name}, self.conversation_id
            agent_spec = self.agent_specs[0]  # Default to general
        self.last_agent_used = agent_spec["name"]
        return None, agent_spec, self.conversation_id

    async def process_message(self, user_input: str) -> Tuple[str, str, int]:
        try:
            error_msg, agent_spec, conv_id = await self._prepare_message(user_input)
            if error_msg:
                return error_msg, agent_spec["name"], conv_id
            # Non-streaming response
            config = {"configurable": {"thread_id": agent_spec["thread_id"]}}
            response = await agent_spec["agent"].ainvoke({"messages": list(self.history)}, config)
            ai_message = response["messages"][-1].content
            # Append assistant reply to history
            self.history.append({"role": "assistant", "content": ai_message})
            return ai_message, agent_spec["name"], self.conversation_id
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            error_message = "I encountered an error while processing your request."
            self.history.append({"role": "assistant", "content": error_message})
            return error_message, "error", self.conversation_id

    async def process_message_stream(self, user_input: str) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream the response token-by-token. Yields dicts like:
          {"type": "agent", "agent": "general"}
          {"type": "token", "content": "Hello"}
          {"type": "token", "content": " world"}
          {"type": "done", "full_response": "Hello world"}
          {"type": "error", "content": "..."}
        """
        try:
            error_msg, agent_spec, conv_id = await self._prepare_message(user_input)
            if error_msg:
                yield {"type": "error", "content": error_msg}
                return

            # Tell the client which agent was selected
            yield {"type": "agent", "agent": agent_spec["name"]}

            config = {"configurable": {"thread_id": agent_spec["thread_id"]}}
            full_response = ""

            async for event in agent_spec["agent"].astream_events(
                {"messages": list(self.history)}, config, version="v2"
            ):
                kind = event.get("event")
                # Stream chat model tokens
                if kind == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        full_response += chunk.content
                        yield {"type": "token", "content": chunk.content}

            # Append full response to history
            if full_response:
                self.history.append({"role": "assistant", "content": full_response})
            yield {"type": "done", "full_response": full_response}

        except Exception as e:
            logger.error(f"Error in streaming: {e}")
            yield {"type": "error", "content": "I encountered an error while processing your request."}


class MosaicAPI:
    def __init__(self, server_configs: List[Dict[str, Any]], web_search: bool = True, model_config=None):
        self.server_configs = server_configs
        self.web_search = web_search
        self.model_config = model_config
        self.mosaic: Optional[Mosaic] = None
        self.conversation_id: Optional[int] = None
        self.user_id: Optional[str] = None
        self.conversation_db = None

    async def initialize(self, conversation_id: Optional[int] = None, user_id: Optional[str] = None):
        # Initialize Mosaic instance asynchronously
        from utils.ConversationDB import ConversationManager
        self.conversation_db = ConversationManager()
        self.conversation_id = conversation_id if conversation_id is not None else -1
        self.user_id = user_id
        self.mosaic = await self._create_mosaic()

    async def _create_mosaic(self):
        # Build and return a Mosaic instance
        agent_specs, inactive_agents = await Mosaic._initialize_agents(self, self.server_configs, self.web_search)
        return Mosaic(agent_specs, inactive_agents, self.server_configs, self.web_search, conversation_id=self.conversation_id, user_id=self.user_id, model_config=self.model_config)

    async def refresh_servers(self) -> Dict[str, Any]:
        """Hot-reload MCP servers. Call this to pick up servers started after boot."""
        if not self.mosaic:
            await self.initialize()
        return await self.mosaic.refresh_mcp_servers()

    async def get_status(self) -> Dict[str, Any]:
        """Get current agent status — which are active, which MCP servers are inactive."""
        if not self.mosaic:
            return {"agents": [], "inactive_servers": []}
        return {
            "agents": [spec["name"] for spec in self.mosaic.agent_specs],
            "inactive_servers": self.mosaic.inactive_agents,
        }

    async def chat(self, message: str, conversation_id: Optional[int] = None, user_id: Optional[str] = None) -> Dict[str, Any]:
        # Public API for non-streaming chat with Mosaic
        if not self.mosaic:
            await self.initialize(conversation_id=conversation_id, user_id=user_id)

        # If conversation changed, reload history from DB
        if conversation_id and conversation_id != self.conversation_id:
            self.conversation_id = conversation_id
            self.mosaic.conversation_id = conversation_id
            # Load prior messages from DB into the agent's history
            if self.conversation_db:
                prior = self.conversation_db.get_conversation_context(conversation_id, max_messages=MAX_HISTORY_EXCHANGES * 2)
                self.mosaic.history.clear()
                # Re-add date context
                today = datetime.datetime.now().strftime('%Y-%m-%d')
                self.mosaic.history.append({
                    "role": "system",
                    "content": (
                        f"For your context only: today's date is {today}. "
                        "Do not mention or repeat the date unless the user explicitly asks. "
                        "All answers must be short and direct."
                    )
                })
                for msg in prior:
                    self.mosaic.history.append(msg)

        response, agent, _ = await self.mosaic.process_message(message)
        return {"response": response, "agent": agent}

    async def chat_stream(self, message: str, conversation_id: Optional[int] = None, user_id: Optional[str] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """Streaming version of chat. Yields SSE-compatible dicts."""
        if not self.mosaic:
            await self.initialize(conversation_id=conversation_id, user_id=user_id)

        # If conversation changed, reload history from DB
        if conversation_id and conversation_id != self.conversation_id:
            self.conversation_id = conversation_id
            self.mosaic.conversation_id = conversation_id
            if self.conversation_db:
                prior = self.conversation_db.get_conversation_context(conversation_id, max_messages=MAX_HISTORY_EXCHANGES * 2)
                self.mosaic.history.clear()
                today = datetime.datetime.now().strftime('%Y-%m-%d')
                self.mosaic.history.append({
                    "role": "system",
                    "content": (
                        f"For your context only: today's date is {today}. "
                        "Do not mention or repeat the date unless the user explicitly asks. "
                        "All answers must be short and direct."
                    )
                })
                for msg in prior:
                    self.mosaic.history.append(msg)

        async for chunk in self.mosaic.process_message_stream(message):
            yield chunk