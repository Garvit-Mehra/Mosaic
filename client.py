#!/usr/bin/env python3
"""
Mosaic - Modular Multi-Agent Tools for Python

A modern toolkit for building, combining, and experimenting with modular multi-agent tools.

Author: Garvit Mehra
Version: 1.1.0
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
from langchain_community.tools.tavily_search import TavilySearchResults

# Local imports
from utils.RAGTools import load_document, query_documents, list_documents, clear_documents

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mosaic.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress httpx info/debug logs everywhere
logging.getLogger("httpx").setLevel(logging.WARNING)

# Configuration
MODEL_NAME = "gpt-4.1-mini"
MAX_HISTORY_EXCHANGES = 5

# API Key validation
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY not set in environment variables. Please check your .env file.")

# --- MCPClientManager for Modular Clients ---
class MCPClientManager:
    def __init__(self, server_configs):
        # Build a dict of server configs for MultiServerMCPClient
        self.server_dict = {}
        for config in server_configs:
            entry = {k: v for k, v in config.items() if k in ("url", "transport", "command", "args")}
            if "transport" not in entry:
                entry["transport"] = "sse"
            self.server_dict[config["name"]] = entry
        # Use a single MultiServerMCPClient for all servers
        self.client = MultiServerMCPClient(self.server_dict)
    def get_client(self):
        return self.client

# Async health check for server
async def is_server_active(url: str) -> bool:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=2) as resp:
                return resp.status == 200
    except Exception:
        return False

# --- Modular async function to get MCP tools for a given server ---
async def get_mcp_tools_modular(client_manager, server_name: str):
    client = client_manager.get_client()
    if not client:
        logger.error(f"No MCP client found for server: {server_name}")
        return []
    try:
        logger.info(f"Attempting to load MCP tools for {server_name}")
        # Use get_tools(server_name=...) for reliability
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
    Usage:
        from client import Mosaic
        mosaic = Mosaic.create(server_configs=SERVER_CONFIGS, web_search=True, rag=True)
        mosaic.run()
    """
    def __init__(self, agent_specs, inactive_agents, server_configs, web_search, rag):
        self.history = deque(maxlen=MAX_HISTORY_EXCHANGES * 2)
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        self.history.append({"role": "system", "content": f"Today's date is {today}."})
        self.last_agent_used = None
        self.classifier_llm = ChatOpenAI(model=MODEL_NAME)
        self.agent_specs = agent_specs
        self.inactive_agents = inactive_agents
        self.server_configs = server_configs
        self.web_search = web_search
        self.rag = rag

    @classmethod
    def create(cls, server_configs: List[Dict[str, Any]], web_search: bool = True, rag: bool = True):
        import asyncio
        async def _acreate():
            agent_specs, inactive_agents = await cls._initialize_agents(server_configs, web_search, rag)
            return cls(agent_specs, inactive_agents, server_configs, web_search, rag)
        return asyncio.run(_acreate())

    def run(self):
        import asyncio
        asyncio.run(self._arun())

    async def _arun(self):
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
                response, agent_name = await self.process_message(user_input)
                display_name = agent_display_names.get(agent_name, agent_name.title())
                print(f"Mosaic: {response}")
            except KeyboardInterrupt:
                print("\n\nSession interrupted. Goodbye!")
                break
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                print(f"\nAn unexpected error occurred: {e}")

    @staticmethod
    async def _initialize_agents(server_configs, web_search, rag) -> Tuple[List[Dict[str, Any]], List[str]]:
        agents = []
        inactive = []
        # General agent (always available)
        agents.append({
            "name": "general",
            "description": "Handles general conversation, chit-chat, and questions that do not require web search or database access. Use this for follow-up questions, clarifications, and general assistance.",
            "agent": create_react_agent(
                ChatOpenAI(model=MODEL_NAME),
                tools=[],
                prompt="You are a helpful assistant for general conversation. If the user asks follow-up questions about previous topics or requests clarification, provide helpful responses based on the conversation context. If you need specific information that was discussed with other agents, ask the user to rephrase or provide more context.",
                checkpointer=MemorySaver()
            ),
            "thread_id": "thread_general"
        })
        # Web agent (optional)
        if web_search:
            agents.append({
                "name": "web",
                "description": "ONLY use for queries that absolutely require current, real-time information from the internet that cannot be answered from general knowledge. Examples: recent news, live sports scores, current weather, stock prices, breaking events. Avoid using for general facts, definitions, or information that should be in your training data.",
                "agent": create_react_agent(
                    ChatOpenAI(model=MODEL_NAME),
                    tools=[TavilySearchResults(api_key=TAVILY_API_KEY, max_results=2)],
                    prompt="You are a web search specialist. Only use web search when absolutely necessary for current, real-time information. For general facts, definitions, or historical information, rely on your training data first. Be selective about when to search the web.",
                    checkpointer=MemorySaver()
                ),
                "thread_id": "thread_web"
            })
        # MCP agents (dynamic)
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
                        prompt=f"You are a specialist for the {config['name'].replace('_', ' ').title()} agent. {config['description']}",
                        checkpointer=MemorySaver()
                    ),
                    "thread_id": f"thread_{config['name']}"
                })
            else:
                logger.warning(f"Server {config['name']} at {config['url']} is inactive or unreachable. Skipping agent.")
                inactive.append(config["name"])
        # RAG agent (optional)
        if rag:
            agents.append({
                "name": "rag",
                "description": "Handles document queries, file processing, and RAG (Retrieval-Augmented Generation) tasks. Use this for questions about PDFs, images, or any loaded documents. Can load, query, and analyze document content.",
                "agent": create_react_agent(
                    ChatOpenAI(model=MODEL_NAME),
                    tools=[load_document, query_documents, list_documents, clear_documents],
                    prompt="You are a document analysis and RAG specialist. Handle queries about PDFs, images, and other documents. Use your tools to load documents, search through them, and provide detailed answers based on document content. When users ask about documents, first check if documents are loaded, then search for relevant information. Be thorough in your document analysis and provide specific information from the documents.",
                    checkpointer=MemorySaver()
                ),
                "thread_id": "thread_rag"
            })
        return agents, inactive

    def _build_classification_prompt(self, user_input: str, conversation_context: deque, last_agent: Optional[str]) -> str:
        prompt = (
            "You are an intelligent intent classifier for a multi-agent system. Analyze the user message and conversation context to select the most appropriate agent.\n\n"
            "GENERAL RULES:\n"
            "1. CONTEXT CONTINUITY: If the user asks a follow-up question related to the previous conversation, use the SAME agent that handled the previous query. Look for:\n"
            "    - Pronouns referring to previous content (them, it, this, that, etc.)\n"
            "    - Short follow-up questions (name them, show me, etc.)\n"
            "    - Questions that build on previous information\n"
            "    - Requests for more details about what was just discussed\n"
        )
        # Add web agent rule if present
        if any(spec['name'] == 'web' for spec in self.agent_specs):
            prompt += (
                "2. WEB AGENT: If the user's query requires current, real-time, or up-to-date information from the internet (such as recent news, live sports scores, current weather, or breaking events), use the web agent. For general facts, definitions, or historical information, prefer the general agent.\n"
            )
        # Add rag agent rule if present
        if any(spec['name'] == 'rag' for spec in self.agent_specs):
            prompt += (
                "3. RAG AGENT: If the user's query is about documents, files, PDFs, images, or content that may be found in loaded documents, use the rag agent.\n"
            )
        prompt += (
            "4. AGENT SELECTION: For each new user message, select the agent whose description best matches the user's intent. Carefully read each agent's description and match the user's request to the most relevant agent.\n\n"
            "5. GENERAL FALLBACK: If you are unsure, prefer the general agent, unless the user's request clearly matches another agent's description.\n\n"
            "Agent options (only active agents are listed):\n"
        )
        for spec in self.agent_specs:
            prompt += f"- {spec['name']}: {spec['description']}\n"
        prompt += f"\nConversation Context:\n"
        if conversation_context:
            recent_context = list(conversation_context)[-6:]
            for msg in recent_context:
                role = "User" if msg["role"] == "user" else "Assistant"
                prompt += f"{role}: {msg['content']}\n"
        prompt += f"\nLast agent used: {last_agent or 'None'}\n"
        prompt += f"Current user message: {user_input}\n\n"
        prompt += "Analyze the conversation flow and determine if this is a follow-up to the previous exchange. Consider the context, pronouns, and whether the user is building on previous information. Respond ONLY with the agent name:"
        return prompt

    async def process_message(self, user_input: str) -> tuple[str, str]:
        try:
            self.history.append({"role": "user", "content": user_input})
            prompt = self._build_classification_prompt(user_input, self.history, self.last_agent_used)
            classification = await self.classifier_llm.ainvoke(prompt)
            agent_name = classification.content.strip().split()[0].lower()
            agent_spec = next((spec for spec in self.agent_specs if spec["name"] == agent_name), None)
            if agent_spec is None:
                if hasattr(self, "inactive_agents") and agent_name in self.inactive_agents:
                    msg = f"Sorry, the {agent_name.replace('_', ' ')} agent is currently unavailable."
                    self.history.append({"role": "assistant", "content": msg})
                    return msg, agent_name
                agent_spec = self.agent_specs[0]
            self.last_agent_used = agent_spec["name"]
            config = {"configurable": {"thread_id": agent_spec["thread_id"]}}
            response = await agent_spec["agent"].ainvoke({"messages": list(self.history)}, config)
            ai_message = response["messages"][-1].content
            self.history.append({"role": "assistant", "content": ai_message})
            return ai_message, agent_spec["name"]
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            error_message = f"I encountered an error while processing your request. Please try again or rephrase your question."
            self.history.append({"role": "assistant", "content": error_message})
            return error_message, "error"

# No main() or hardcoded configs. Use Mosaic as a library from another script.
