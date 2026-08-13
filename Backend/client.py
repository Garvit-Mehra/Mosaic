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
from typing import List, Dict, Any, Optional, AsyncGenerator

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
    logger.warning("TAVILY_API_KEY not set in environment variables. Web search tools will not be available.")


# =============================================================================
# MCP Server Utilities
# =============================================================================



async def is_server_active(url: Optional[str]) -> bool:
    if not url:
        return True  # stdio servers (no url) are assumed active, will fail at connection if not
    try:
        # Skip SSL verification for external servers (some have cert issues on macOS)
        connector = aiohttp.TCPConnector(ssl=False)
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                # Any response means the server is reachable and listening
                return True
    except Exception:
        return False


async def get_mcp_tools(server_name: str, config: Dict[str, Any]):
    default_transport = "sse" if config.get("url") else "stdio"
    transport = config.get("transport") or default_transport
    
    client_config = {k: v for k, v in config.items() if k in ("url", "command", "args") and v is not None}
    client_config["transport"] = transport

    try:
        client = MultiServerMCPClient({server_name: client_config})
        tools = await client.get_tools(server_name=server_name)
        config["transport"] = transport  # Store working transport
        logger.info(f"Loaded {len(tools)} tools for {server_name} via {transport}")
        return tools
    except Exception as e:
        if transport == "sse":
            logger.warning(f"Transport sse failed for {server_name}: {e}. Trying http fallback...")
            try:
                client_config["transport"] = "http"
                client = MultiServerMCPClient({server_name: client_config})
                tools = await client.get_tools(server_name=server_name)
                config["transport"] = "http"  # Store working transport
                logger.info(f"Loaded {len(tools)} tools for {server_name} via http fallback")
                return tools
            except Exception as e2:
                logger.error(f"Both sse and http transports failed for {server_name}: {e2}")
                return []
        else:
            logger.error(f"Transport {transport} failed for {server_name}: {e}")
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
                    tools=[TavilySearch(api_key=TAVILY_API_KEY, max_results=3)],
                    prompt=(
                        "You are Mosaic, a helpful AI assistant with access to web search.\n"
                        "When you receive a query:\n"
                        "1. Use the search tool to find current information.\n"
                        "2. Read the search results carefully.\n"
                        "3. Write a clear, natural language answer as if explaining to a person.\n"
                        "4. Do NOT dump raw search results, JSON, or URLs at the user.\n"
                        "5. Do NOT say 'According to search results' or 'Based on the results' — just answer directly.\n"
                        "6. Include specific facts, numbers, and dates from the results.\n"
                        "7. Keep it concise — 2-4 sentences unless more detail is needed.\n"
                        "8. If the search returns nothing useful, say so honestly.\n"
                        "9. If a search tool fails, encounters an error, or needs to be retried, DO NOT output any apology or explanation (e.g. do NOT say 'It seems that there was an issue...'). Simply call the tool again immediately and silently."
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
                for config in active_configs:
                    try:
                        mcp_tools = await get_mcp_tools(config["name"], config)
                        agents.append({
                            "name": config["name"],
                            "description": config["description"],
                            "tools": mcp_tools,
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
            for config in servers_to_check:
                if await is_server_active(config.get("url")):
                    try:
                        mcp_tools = await get_mcp_tools(config["name"], config)
                        
                        # Remove existing agent with the same name if it exists to avoid duplicates
                        self.agents = [a for a in self.agents if a["name"] != config["name"]]
                        
                        self.agents.append({
                            "name": config["name"],
                            "description": config["description"],
                            "tools": mcp_tools,
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
            if config and not await is_server_active(config.get("url")):
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
        # Build agent list — only include non-general agents with descriptions
        mcp_agents = [a for a in self.registry.agents if a["name"] not in ("general", "web", "rag")]

        prompt = (
            "You are a request router. Output ONLY a single agent name — nothing else.\n\n"
            "ROUTING RULES (apply in order):\n"
            "1. Output 'web' ONLY if the query explicitly requires LIVE or REAL-TIME data that changes daily: "
            "current news headlines, live sports scores, today's stock prices, current weather. "
            "Do NOT use 'web' for general knowledge, definitions, explanations, history, or anything that doesn't change.\n"
            "2. Output 'rag' ONLY if the user explicitly says 'my document', 'the PDF', 'loaded file', "
            "or directly references content they uploaded. Do NOT use 'rag' for general questions.\n"
        )

        if mcp_agents:
            prompt += "3. Output an MCP agent name ONLY if the query explicitly requires a tool from that server:\n"
            for a in mcp_agents:
                prompt += f"   - '{a['name']}': {a['description']}\n"
            prompt += "4. For EVERYTHING ELSE — coding, math, writing, explanations, opinions, creative tasks, follow-up questions — output 'general'.\n"
        else:
            prompt += "3. For EVERYTHING ELSE — output 'general'.\n"

        prompt += (
            "\nExamples:\n"
            "- 'what is a binary tree?' → general\n"
            "- 'write me a python function' → general\n"
            "- 'what happened in the news today?' → web\n"
            "- 'what is the current price of bitcoin?' → web\n"
            "- 'what does my report say about revenue?' → rag\n"
            "- 'hello' → general\n"
            "- 'explain recursion' → general\n"
            "\n"
            f"User query: {user_input}\n\n"
            "Output ONLY the agent name. No explanation, no punctuation."
        )

        result = await self.classifier.ainvoke(prompt)
        name = result.content.strip().split()[0].lower().strip(".,!?")

        # Validate — default to general if unrecognised
        if not self.registry.get_agent(name):
            if name in self.registry.inactive_servers:
                return "__inactive__:" + name
            return "general"
        return name

    async def chat(self, message: str, conversation_id: Optional[int] = None, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Process a chat message (non-streaming). Stateless.
        
        Detects long-running operations and offloads them to the FlowQ,
        returning an immediate status message instead of blocking.
        """
        # Check if this is a background-eligible operation
        bg_result = await self._check_background_task(message, user_id)
        if bg_result:
            return bg_result

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

    async def _check_background_task(self, message: str, user_id: Optional[str]) -> Optional[Dict[str, Any]]:
        """
        Detect if a message requires a long-running background operation.
        If so, submit it to the FlowQ and return an immediate response.
        Returns None if the message should be processed normally.
        """
        msg_lower = message.lower().strip()

        try:
            from flowq.src.mosaic_bridge import submit_background_job
        except ImportError:
            # FlowQ not available — process everything synchronously
            return None

        # Pattern: "load/process/index [file path]"
        # Detect RAG document ingestion requests
        rag_keywords = ["load document", "process pdf", "index file", "ingest"]
        for keyword in rag_keywords:
            if keyword in msg_lower:
                # Extract file path (naive: take the quoted or last segment)
                import re
                path_match = re.search(r'["\']([^"\']+)["\']', message) or re.search(r'(\S+\.(pdf|png|jpg|jpeg|txt|docx))(\s|$)', msg_lower)
                if path_match:
                    file_path = path_match.group(1)
                    job_id = await submit_background_job(
                        job_type="rag_process",
                        payload={"file_path": file_path},
                        user_id=user_id or "anonymous",
                        priority=5,
                        timeout_seconds=600,
                    )
                    return {
                        "response": f"Processing document in the background (job: {job_id}). I'll have it ready for questions shortly. Check status with: GET /jobs/{job_id}",
                        "agent": "system",
                        "job_id": job_id,
                    }

        # Pattern: "scrape/fetch/read [URL]"
        scrape_keywords = ["scrape", "fetch url", "read from http", "ingest url", "load url"]
        for keyword in scrape_keywords:
            if keyword in msg_lower:
                import re
                url_match = re.search(r'(https?://\S+)', message)
                if url_match:
                    url = url_match.group(1)
                    job_id = await submit_background_job(
                        job_type="web_scrape",
                        payload={"url": url, "max_length": 20000},
                        user_id=user_id or "anonymous",
                        priority=5,
                        timeout_seconds=60,
                    )
                    return {
                        "response": f"Fetching content from {url} in the background (job: {job_id}). Check status with: GET /jobs/{job_id}",
                        "agent": "system",
                        "job_id": job_id,
                    }

        # Pattern: check job status ("job status", "is my job done", etc.)
        status_keywords = ["job status", "check job", "is my job", "is it done", "job progress"]
        for keyword in status_keywords:
            if keyword in msg_lower:
                import re
                from flowq.src.mosaic_bridge import get_job_result
                # Look for a UUID-like pattern
                uuid_match = re.search(r'([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})', msg_lower)
                if uuid_match:
                    job_id = uuid_match.group(1)
                    result = await get_job_result(job_id)
                    status = result.get("status", "unknown")
                    if status == "completed":
                        job_result = result.get("result", {})
                        return {
                            "response": f"Job {job_id} is complete.\n\nResult: {job_result}",
                            "agent": "system",
                        }
                    elif status == "failed":
                        error = result.get("error", "Unknown error")
                        return {
                            "response": f"Job {job_id} failed: {error}",
                            "agent": "system",
                        }
                    else:
                        return {
                            "response": f"Job {job_id} is currently: {status}",
                            "agent": "system",
                        }

        return None

    async def chat_stream(self, message: str, conversation_id: Optional[int] = None, user_id: Optional[str] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """Process a chat message with streaming. Stateless.
        
        Detects background tasks and yields an immediate response if offloaded.
        """
        # Check if this should be a background task
        bg_result = await self._check_background_task(message, user_id)
        if bg_result:
            yield {"type": "agent", "agent": "system"}
            yield {"type": "token", "content": bg_result["response"]}
            yield {"type": "done", "full_response": bg_result["response"]}
            return

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
