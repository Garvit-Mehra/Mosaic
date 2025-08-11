#!/usr/bin/env python3
"""
Mosaic - Modular Multi-Agent Tools for Python

A modern toolkit for building, combining, and experimenting with modular multi-agent tools.

Author: Garvit Mehra
Version: 1.2.0
License: MIT
"""

import os
import asyncio
import logging
import aiohttp
import datetime
from collections import deque
from typing import List, Dict, Any, Optional, Tuple

# Third-party imports
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_tavily import TavilySearch

# Local imports
from utils.RAGTools import load_document, query_documents, list_documents, clear_documents

# Load environment variables from .env file
load_dotenv()

# Configure logging to file (INFO level by default)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mosaic.log'),
    ]
)
logger = logging.getLogger(__name__)

# Suppress verbose HTTP logs from `httpx`
logging.getLogger("httpx").setLevel(logging.WARNING)

# Core configuration constants
MODEL_NAME = "gpt-5-nano"
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
    """
    def __init__(self, agent_specs, inactive_agents, server_configs, web_search):
        # Conversation history (system + user + assistant messages)
        self.history = deque(maxlen=MAX_HISTORY_EXCHANGES * 2)
        # Add system date context
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        self.history.append({"role": "system", "content": f"Today's date is {today}."})
        self.last_agent_used = None
        self.classifier_llm = ChatOpenAI(model=MODEL_NAME)  # Used for agent selection
        self.agent_specs = agent_specs
        self.inactive_agents = inactive_agents
        self.server_configs = server_configs
        self.web_search = web_search

    @classmethod
    def create(cls, server_configs: List[Dict[str, Any]], web_search: bool = True):
        # Async wrapper to allow synchronous entrypoint
        import asyncio
        async def _acreate():
            agent_specs, inactive_agents = await cls._initialize_agents(server_configs, web_search)
            return cls(agent_specs, inactive_agents, server_configs, web_search)
        return asyncio.run(_acreate())

    def run(self):
        # Main loop (blocking) for CLI mode
        asyncio.run(self._arun())

    async def _arun(self):
        # CLI interface with live agent routing
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
                response, agent_name = await self.process_message(user_input)
                display_name = agent_display_names.get(agent_name, agent_name.title())
                print(f"Mosaic ({display_name}): {response}")
            except KeyboardInterrupt:
                print("\n\nSession interrupted. Goodbye!")
                break
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                print(f"\nAn unexpected error occurred: {e}")

    @staticmethod
    async def _initialize_agents(server_configs, web_search) -> Tuple[List[Dict[str, Any]], List[str]]:
        agents = []
        inactive = []

        # Always-available general agent
        agents.append({
            "name": "general",
            "description": "For general conversation and follow-ups.",
            "agent": create_react_agent(
                ChatOpenAI(model=MODEL_NAME),
                tools=[],
                prompt="You are a general assistant in Mosaic...",
                checkpointer=MemorySaver()
            ),
            "thread_id": "thread_general"
        })

        # Optional web search agent
        if web_search:
            agents.append({
                "name": "web",
                "description": "For real-time internet data queries.",
                "agent": create_react_agent(
                    ChatOpenAI(model=MODEL_NAME),
                    tools=[TavilySearch(api_key=TAVILY_API_KEY, max_results=2)],
                    prompt="You are a web specialist in Mosaic...",
                    checkpointer=MemorySaver()
                ),
                "thread_id": "thread_web"
            })

        # MCP agents from server configs
        client_manager = MCPClientManager(server_configs)
        for config in server_configs:
            if await is_server_active(config["url"]):
                mcp_tools = await get_mcp_tools_modular(client_manager, config["name"])
                agents.append({
                    "name": config["name"],
                    "description": config["description"],
                    "agent": create_react_agent(
                        ChatOpenAI(model=MODEL_NAME),
                        tools=mcp_tools,
                        prompt=f"You are the {config['name']} specialist in Mosaic...",
                        checkpointer=MemorySaver()
                    ),
                    "thread_id": f"thread_{config['name']}"
                })
            else:
                logger.warning(f"Server {config['name']} at {config['url']} is inactive or unreachable. Skipping agent.")
                inactive.append(config["name"])

        # RAG agent for document queries
        agents.append({
            "name": "rag",
            "description": "For document and RAG queries.",
            "agent": create_react_agent(
                ChatOpenAI(model=MODEL_NAME),
                tools=[load_document, query_documents, list_documents, clear_documents],
                prompt="You are a document specialist in Mosaic...",
                checkpointer=MemorySaver()
            ),
            "thread_id": "thread_rag"
        })

        return agents, inactive

    def _build_classification_prompt(self, user_input: str, conversation_context: deque, last_agent: Optional[str]) -> str:
        # Builds the prompt for the classifier LLM to select the best agent
        prompt = (
            "Classify intent for Mosaic, routing queries to agents..."
        )
        if any(spec['name'] == 'web' for spec in self.agent_specs):
            prompt += "2. Web agent for real-time data.\n"
        prompt += "3. RAG agent for document queries.\n"
        for spec in self.agent_specs:
            prompt += f"- {spec['name']}: {spec['description']}\n"
        # Add recent conversation context
        if conversation_context:
            for msg in list(conversation_context)[-6:]:
                prompt += f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}\n"
        prompt += f"Last agent: {last_agent or 'None'}\n"
        prompt += f"Query: {user_input}\n\n"
        prompt += "Return only the agent name:"
        return prompt

    async def process_message(self, user_input: str) -> tuple[str, str]:
        try:
            # Append user message to history
            self.history.append({"role": "user", "content": user_input})
            # Build classification prompt
            prompt = self._build_classification_prompt(user_input, self.history, self.last_agent_used)
            # Get classification from LLM
            classification = await self.classifier_llm.ainvoke(prompt)
            agent_name = classification.content.strip().split()[0].lower()

            # Lookup matching agent spec
            agent_spec = next((spec for spec in self.agent_specs if spec["name"] == agent_name), None)

            # Handle missing agent
            if agent_spec is None:
                if hasattr(self, "inactive_agents") and agent_name in self.inactive_agents:
                    msg = f"Sorry, the {agent_name.replace('_', ' ')} agent is currently unavailable."
                    self.history.append({"role": "assistant", "content": msg})
                    return msg, agent_name
                agent_spec = self.agent_specs[0]  # Default to general

            # Store last used agent for follow-up tracking
            self.last_agent_used = agent_spec["name"]

            # Prepare agent invocation config
            config = {"configurable": {"thread_id": agent_spec["thread_id"]}}
            response = await agent_spec["agent"].ainvoke({"messages": list(self.history)}, config)
            ai_message = response["messages"][-1].content

            # Append assistant reply to history
            self.history.append({"role": "assistant", "content": ai_message})
            return ai_message, agent_spec["name"]

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            error_message = "I encountered an error while processing your request."
            self.history.append({"role": "assistant", "content": error_message})
            return error_message, "error"
