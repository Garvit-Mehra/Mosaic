#!/usr/bin/env python3
"""
Mosaic - Modular Multi-Agent Tools for Python

A modern toolkit for building, combining, and experimenting with modular multi-agent tools.

Author: Garvit Mehra
Version: 1.0.0
License: MIT
"""

import os
import asyncio
import logging
from collections import deque
from typing import List, Dict, Any, Optional

# Third-party imports
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_mcp_adapters.client import MultiServerMCPClient

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

# Configuration
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8000/sse")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
MAX_HISTORY_EXCHANGES = int(os.getenv("MAX_HISTORY_EXCHANGES", "5"))

# API Key validation
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY not set in environment variables. Please check your .env file.")

# Initialize MCP client
mcp_client = MultiServerMCPClient({
    "local": {
        "url": MCP_SERVER_URL,
        "transport": "sse",
    }
})


class Mosaic:
    """
    Main Mosaic client class that manages the multi-agent system.
    """
    
    def __init__(self):
        """Initialize the Mosaic system with all agents and tools."""
        self.history = deque(maxlen=MAX_HISTORY_EXCHANGES * 2)  # 2 messages per exchange
        self.last_agent_used = None
        self.classifier_llm = ChatOpenAI(model=MODEL_NAME)
        self.agent_specs = self._initialize_agents()
        
    def _initialize_agents(self) -> List[Dict[str, Any]]:
        """
        Initialize all specialized agents with their tools and configurations.
        
        Returns:
            List of agent specifications
        """
        return [
            {
                "name": "general",
                "description": "Handles general conversation, chit-chat, and questions that do not require web search or database access. Use this for follow-up questions, clarifications, and general assistance.",
                "agent": create_react_agent(
                    ChatOpenAI(model=MODEL_NAME),
                    tools=[],
                    prompt="You are a helpful assistant for general conversation. If the user asks follow-up questions about previous topics or requests clarification, provide helpful responses based on the conversation context. If you need specific information that was discussed with other agents, ask the user to rephrase or provide more context.",
                    checkpointer=MemorySaver()
                ),
                "thread_id": "thread_general"
            },
            {
                "name": "web",
                "description": "ONLY use for queries that absolutely require current, real-time information from the internet that cannot be answered from general knowledge. Examples: recent news, live sports scores, current weather, stock prices, breaking events. Avoid using for general facts, definitions, or information that should be in your training data.",
                "agent": create_react_agent(
                    ChatOpenAI(model=MODEL_NAME),
                    tools=[TavilySearchResults(api_key=TAVILY_API_KEY, max_results=2)],
                    prompt="You are a web search specialist. Only use web search when absolutely necessary for current, real-time information. For general facts, definitions, or historical information, rely on your training data first. Be selective about when to search the web.",
                    checkpointer=MemorySaver()
                ),
                "thread_id": "thread_web"
            },
            {
                "name": "db_manager",
                "description": "Handles all database-related queries including viewing, editing, querying tables and records, data analysis, and follow-up questions about database content. Use this for any question about stored data, records, or database operations.",
                "agent": create_react_agent(
                    ChatOpenAI(model=MODEL_NAME),
                    tools=asyncio.run(self._get_mcp_tools()),
                    prompt="You are a database management specialist. Handle all database queries, data analysis, and follow-up questions about database content. If the user asks follow-up questions about data you've previously retrieved, use your tools to get the specific information they need. Be thorough in providing database information and analysis.",
                    checkpointer=MemorySaver()
                ),
                "thread_id": "thread_db"
            },
            {
                "name": "rag",
                "description": "Handles document queries, file processing, and RAG (Retrieval-Augmented Generation) tasks. Use this for questions about PDFs, images, or any loaded documents. Can load, query, and analyze document content.",
                "agent": create_react_agent(
                    ChatOpenAI(model=MODEL_NAME),
                    tools=[load_document, query_documents, list_documents, clear_documents],
                    prompt="You are a document analysis and RAG specialist. Handle queries about PDFs, images, and other documents. Use your tools to load documents, search through them, and provide detailed answers based on document content. When users ask about documents, first check if documents are loaded, then search for relevant information. Be thorough in your document analysis and provide specific information from the documents.",
                    checkpointer=MemorySaver()
                ),
                "thread_id": "thread_rag"
            }
        ]
    
    async def _get_mcp_tools(self):
        """Get MCP tools for database operations."""
        try:
            return await mcp_client.get_tools()
        except Exception as e:
            logger.warning(f"Failed to get MCP tools: {e}")
            return []
    
    def _build_classification_prompt(self, user_input: str, conversation_context: deque, last_agent: Optional[str]) -> str:
        """
        Build a comprehensive prompt for intent classification.
        
        Args:
            user_input: The current user message
            conversation_context: Recent conversation history
            last_agent: The last agent used
            
        Returns:
            Formatted classification prompt
        """
        prompt = """You are an intelligent intent classifier for a multi-agent system. Analyze the user message and conversation context to select the most appropriate agent.

IMPORTANT RULES:
1. CONTEXT CONTINUITY: If the user asks a follow-up question related to the previous conversation, use the SAME agent that handled the previous query. Look for:
   - Pronouns referring to previous content (them, it, this, that, etc.)
   - Short follow-up questions (name them, show me, etc.)
   - Questions that build on previous information
   - Requests for more details about what was just discussed

2. WEB AGENT RESTRICTION: Only use the web agent for queries that absolutely require current, real-time information (news, live scores, current weather, etc.). For general facts, definitions, or historical information, use the general agent.

3. DATABASE CONTEXT: If the user asks about data, records, or follows up on database queries, use the db_manager agent.

4. DOCUMENT CONTEXT: If the user asks about PDFs, images, documents, or file content, use the rag agent.

5. GENERAL FALLBACK: When in doubt, prefer the general agent over the web agent, unless explicitly asked to search the web.

Agent options:
"""
        for spec in self.agent_specs:
            prompt += f"- {spec['name']}: {spec['description']}\n"
        
        prompt += f"\nConversation Context:\n"
        if conversation_context:
            recent_context = list(conversation_context)[-6:]  # Last 6 messages
            for msg in recent_context:
                role = "User" if msg["role"] == "user" else "Assistant"
                prompt += f"{role}: {msg['content']}\n"
        
        prompt += f"\nLast agent used: {last_agent or 'None'}\n"
        prompt += f"Current user message: {user_input}\n\n"
        prompt += "Analyze the conversation flow and determine if this is a follow-up to the previous exchange. Consider the context, pronouns, and whether the user is building on previous information. Respond ONLY with the agent name:"
        return prompt
    
    async def process_message(self, user_input: str) -> str:
        """
        Process a user message through the multi-agent system.
        
        Args:
            user_input: The user's message
            
        Returns:
            The AI response
        """
        try:
            # Add user message to history
            self.history.append({"role": "user", "content": user_input})
            
            # Classify intent
            prompt = self._build_classification_prompt(user_input, self.history, self.last_agent_used)
            classification = await self.classifier_llm.ainvoke(prompt)
            agent_name = classification.content.strip().split()[0].lower()
            
            # Find the appropriate agent
            agent_spec = next(
                (spec for spec in self.agent_specs if spec["name"] == agent_name), 
                self.agent_specs[0]  # Default to general agent
            )
            
            self.last_agent_used = agent_spec["name"]
            
            # Route to selected agent
            config = {"configurable": {"thread_id": agent_spec["thread_id"]}}
            response = await agent_spec["agent"].ainvoke({"messages": list(self.history)}, config)
            ai_message = response["messages"][-1].content
            
            # Add agent response to history
            self.history.append({"role": "assistant", "content": ai_message})
            
            return ai_message
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            error_message = f"I encountered an error while processing your request. Please try again or rephrase your question."
            self.history.append({"role": "assistant", "content": error_message})
            return error_message
    
    async def run(self):
        """Start the interactive chat session."""
        print("Mosaic - Modular Multi-Agent Tools for Python")
        print("=" * 50)
        print("Type 'exit' or 'quit' to end the session")
        print("Available agents: General, Web, Database Manager, RAG")
        print("=" * 50)
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in {"exit", "quit"}:
                    print("\nGoodbye! Thank you for using Mosaic.")
                    break
                
                if not user_input:
                    continue
                
                # Process the message
                response = await self.process_message(user_input)
                print(f"\n{self.last_agent_used.capitalize()} Agent: {response}")
                
            except KeyboardInterrupt:
                print("\n\nSession interrupted. Goodbye!")
                break
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                print(f"\nAn unexpected error occurred: {e}")


async def main():
    """Main entry point for Mosaic."""
    try:
        mosaic = Mosaic()
        await mosaic.run()
    except Exception as e:
        logger.error(f"Failed to start Mosaic: {e}")
        print(f"Failed to start Mosaic: {e}")
        print("Please check your configuration and try again.")


if __name__ == "__main__":
    asyncio.run(main())
