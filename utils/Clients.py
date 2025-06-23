import asyncio
import json
import os
import requests
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional, Union
from autogen import AssistantAgent, UserProxyAgent, ConversableAgent
from openai import AsyncOpenAI, OpenAI
from dotenv import load_dotenv
from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters

load_dotenv()

# Tavily API setup
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
TAVILY_URL = "https://api.tavily.com/search"
TAVILY_HEADERS = {
    "Authorization": f"Bearer {TAVILY_API_KEY}",
    "Content-Type": "application/json"
}

class MCPOpenAIClient:
    """Client for interacting with OpenAI models using MCP tools."""

    def __init__(self, model: str = "gpt-4.1-nano", custom_tools: Optional[List[Dict[str, Any]]] = None):
        """Initialize the OpenAI MCP client.

        Args:
            model: The OpenAI model to use.
            custom_tools: Optional list of tools in OpenAI format to be used alongside MCP tools.
        """
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.openai_client = AsyncOpenAI()
        self.model = model
        self.stdio: Optional[Any] = None
        self.stdin: Optional[Any] = None
        self.conversation_history = []
        self.custom_tools = custom_tools or []
        self.system_prompt = """You are Jarvis, a helpful and friendly AI assistant. You have access to various tools to help users manage and interact with their SQLite database and search the web. 
        Always maintain a conversational tone and provide clear, helpful responses. When using tools, explain what you're doing in a natural way.
        If you encounter any errors, explain them clearly and suggest solutions."""

    async def connect_to_server(self, server_script_path: str = "server.py", retries: int = 3, delay: float = 2.0):
        """Connect to an MCP server with retries.

        Args:
            server_script_path: Path to the server script.
            retries: Number of retry attempts.
            delay: Delay between retries in seconds.
        """
        for attempt in range(retries):
            try:
                server_params = StdioServerParameters(
                    command="python",
                    args=[server_script_path],
                )
                stdio_transport = await self.exit_stack.enter_async_context(
                    stdio_client(server_params)
                )
                self.stdio, self.stdin = stdio_transport
                self.session = await self.exit_stack.enter_async_context(
                    ClientSession(self.stdio, self.stdin)
                )
                await self.session.initialize()
                return
            except Exception as e:
                if attempt < retries - 1:
                    await asyncio.sleep(delay)
                else:
                    raise

    async def web_search(self, query: str) -> str:
        """Search the web using Tavily API."""
        payload = {
            "query": query,
            "topic": "general",
            "search_depth": "basic",
            "chunks_per_source": 3,
            "max_results": 3
        }
        try:
            response = requests.post(TAVILY_URL, json=payload, headers=TAVILY_HEADERS)
            data = response.json()
            return json.dumps(data, indent=2)
        except Exception as e:
            return f"Error calling Tavily API: {str(e)}"

    async def get_mcp_tools(self) -> List[Dict[str, Any]]:
        """Get available tools from the MCP server in OpenAI format and combine with custom tools.

        Returns:
            A list of tools in OpenAI format.
        """
        try:
            tools_result = await self.session.list_tools()
            mcp_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "required": []
                        },
                    },
                } for tool in tools_result.tools
            ]
            # Add web search tool
            web_search_tool = {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web using Tavily and return summarized news or information related to the query.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The topic or question to search about."
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
            return mcp_tools + self.custom_tools + [web_search_tool]
        except Exception as e:
            raise

    async def process_query(self, query: str, tool_args: Optional[Dict] = None) -> str:
        """Process a query using OpenAI and available MCP tools.

        Args:
            query: The user query.
            tool_args: Optional arguments for the tool (e.g., filter parameters).

        Returns:
            The response from OpenAI.
        """
        try:
            tools = await self.get_mcp_tools()
            
            if not self.conversation_history:
                self.conversation_history.append({"role": "system", "content": self.system_prompt})
            
            # Add user query to history
            self.conversation_history.append({"role": "user", "content": query})
            
            if tool_args:
                self.conversation_history.append({"role": "user", "content": f"Tool arguments: {json.dumps(tool_args)}"})

            response = await self.openai_client.chat.completions.create(
                model=self.model,
                messages=self.conversation_history,
                tools=tools,
                tool_choice="auto",
            )
            assistant_message = response.choices[0].message
            self.conversation_history.append(assistant_message)

            if assistant_message.tool_calls:
                for tool_call in assistant_message.tool_calls:
                    args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else tool_args or {}
                    
                    # Handle web search tool
                    if tool_call.function.name == "web_search":
                        result = await self.web_search(args.get("query", ""))
                    else:
                        result = await self.session.call_tool(
                            tool_call.function.name,
                            arguments=args,
                        )
                        result = result.content[0].text

                    tool_response = {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    }
                    self.conversation_history.append(tool_response)
                
                final_response = await self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=self.conversation_history,
                    tools=tools,
                    tool_choice="none",
                )
                response_content = final_response.choices[0].message.content
                self.conversation_history.append({"role": "assistant", "content": response_content})
                return response_content

            return assistant_message.content
        except Exception as e:
            error_message = f"I encountered an error while processing your request: {str(e)}"
            self.conversation_history.append({"role": "assistant", "content": error_message})
            return error_message

    async def cleanup(self):
        """Clean up resources and shut down transports safely."""
        try:
            await self.exit_stack.aclose()
            if self.stdio:
                self.stdio.close()
            if self.stdin:
                self.stdin.close()
            self.conversation_history = []
            await asyncio.sleep(0.2)
        except Exception as e:
            print(f"Cleanup error: {e}")

class AutogenMCPClient:
    """Client that uses Autogen agents for interacting with the MCP server."""
    
    def __init__(
        self,
        name: str = "Assistant",
        model: str = "gpt-4.1-nano",
        server_script_path: str = "server.py",
        system_message: Optional[str] = None
    ):
        """Initialize the AutogenMCPClient."""
        self.openai_client = AsyncOpenAI()
        self.model = model
        self.server_script_path = server_script_path
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.is_connected = False
        
        if system_message is None:
            system_message = """You are a helpful AI assistant with tools to manage a SQLite database and search the web.
            You can help users create, view, edit, and manage their database tables and data, as well as search for information online.
            Always maintain a conversational tone and provide clear, helpful responses.
            If you encounter any errors, explain them clearly and suggest solutions."""

        self.agent = AssistantAgent(
            name=name,
            system_message=system_message,
            llm_config={
                "config_list": [{"model": model}],
                "temperature": 0.7,
                "timeout": 60
            }
        )
        
        self.conversation_history = [
            {"role": "system", "content": system_message}
        ]

    async def connect(self):
        """Connect to the MCP server."""
        if not self.is_connected:
            server_params = StdioServerParameters(
                command="python",
                args=[self.server_script_path],
            )
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            self.stdio, self.stdin = stdio_transport
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(self.stdio, self.stdin)
            )
            await self.session.initialize()
            self.is_connected = True

    async def web_search(self, query: str) -> str:
        """Search the web using Tavily API."""
        payload = {
            "query": query,
            "topic": "general",
            "search_depth": "basic",
            "chunks_per_source": 3,
            "max_results": 3
        }
        try:
            response = requests.post(TAVILY_URL, json=payload, headers=TAVILY_HEADERS)
            data = response.json()
            return json.dumps(data, indent=2)
        except Exception as e:
            return f"Error calling Tavily API: {str(e)}"

    async def get_mcp_tools(self) -> List[Dict[str, Any]]:
        """Get available tools from the MCP server."""
        if not self.is_connected:
            await self.connect()
        tools_result = await self.session.list_tools()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    },
                },
            } for tool in tools_result.tools
        ]
        # Add web search tool
        web_search_tool = {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web using Tavily and return summarized news or information related to the query.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The topic or question to search about."
                        }
                    },
                    "required": ["query"]
                }
            }
        }
        return tools + [web_search_tool]

    async def process_query(self, query: str) -> str:
        """Process a query using the Autogen agent and MCP tools."""
        if not self.is_connected:
            await self.connect()

        tools = await self.get_mcp_tools()
        self.conversation_history.append({"role": "user", "content": query})
        
        response = await self.openai_client.chat.completions.create(
            model=self.model,
            messages=self.conversation_history,
            tools=tools,
            tool_choice="auto",
            temperature=0.7,
            max_tokens=1000
        )
        
        assistant_message = response.choices[0].message
        self.conversation_history.append(assistant_message)

        if assistant_message.tool_calls:
            for tool_call in assistant_message.tool_calls:
                args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
                
                # Handle web search tool
                if tool_call.function.name == "web_search":
                    result = await self.web_search(args.get("query", ""))
                else:
                    result = await self.session.call_tool(
                        tool_call.function.name,
                        arguments=args,
                    )
                    result = result.content[0].text

                tool_response = {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
                self.conversation_history.append(tool_response)
            
            final_response = await self.openai_client.chat.completions.create(
                model=self.model,
                messages=self.conversation_history,
                tools=tools,
                tool_choice="none",
            )
            response_content = final_response.choices[0].message.content
            self.conversation_history.append({"role": "assistant", "content": response_content})
            return response_content

        return assistant_message.content

    async def cleanup(self):
        """Clean up resources."""
        try:
            await self.exit_stack.aclose()
            if hasattr(self, 'stdio'):
                self.stdio.close()
            if hasattr(self, 'stdin'):
                self.stdin.close()
            self.conversation_history = []
            self.is_connected = False
            await asyncio.sleep(0.1)
        except Exception as e:
            print(f"Cleanup error: {e}")

    async def chat(self, max_turns: int = 10):
        """Start an interactive chat session.
        
        Args:
            max_turns: Maximum number of conversation turns
        """
        
        turn = 0
        while turn < max_turns:
            # Get user input
            query = input("You: ").strip()
            
            # Check for exit command
            if query.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break

            try:
                response = await self.process_query(query)
                print(f"\nAssistant: {response}\n")
                turn += 1
            except Exception as e:
                print(f"Error: {str(e)}")
                break

class WebSearchClient(AssistantAgent):
    """An Autogen assistant that can search the web."""
    
    def __init__(
        self,
        name: str = "WebSearchAssistant",
        system_message: Optional[str] = None,
        model: str = "gpt-4.1-nano",
    ):
        if system_message is None:
            system_message = """You are a helpful AI assistant that can search the web for information. 
            You have access to real-time web search capabilities through the Tavily API.
            When asked a question, you will search the web if needed and provide up-to-date information."""

        super().__init__(
            name=name,
            system_message=system_message,
            llm_config={
                "config_list": [{"model": model}],
                "temperature": 0.7,
                "timeout": 60
            }
        )

        self.openai_client = AsyncOpenAI()
        self.model = model
        self.register_function(
            function_map={
                "web_search": self.web_search
            }
        )

    async def web_search(self, query: str) -> str:
        """Search the web using Tavily API."""
        payload = {
            "query": query,
            "topic": "general",
            "search_depth": "basic",
            "chunks_per_source": 3,
            "max_results": 3
        }
        try:
            response = requests.post(TAVILY_URL, json=payload, headers=TAVILY_HEADERS)
            data = response.json()
            return json.dumps(data, indent=2)
        except Exception as e:
            return f"Error calling Tavily API: {str(e)}"

    async def run(self, task: str) -> Dict[str, Any]:
        """Run a task using the web search capability."""
        messages = []
        messages.append({"role": "user", "content": task})
        
        response = await self.openai_client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=[{
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web using Tavily and return summarized information.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query"
                            }
                        },
                        "required": ["query"]
                    }
                }
            }],
            tool_choice="auto"
        )
        
        assistant_message = response.choices[0].message
        messages.append(assistant_message)

        if assistant_message.tool_calls:
            for tool_call in assistant_message.tool_calls:
                args = json.loads(tool_call.function.arguments)
                result = await self.web_search(args.get("query", ""))
                
                tool_response = {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                }
                messages.append(tool_response)
            
            final_response = await self.openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=[],
                tool_choice="none"
            )
            messages.append(final_response.choices[0].message)
        
        return {"messages": messages}

class MCPToolSchemaAdapter:
    """
    Adapter to convert MCP tools to OpenAI-compatible tool schemas for use in autogen agents.
    """
    @staticmethod
    def to_openai_schema(mcp_tools) -> list:
        """
        Convert a list of MCP tool objects (from mcp_server_tools or McpWorkbench.list_tools())
        to a list of OpenAI-compatible tool schemas.
        Args:
            mcp_tools: List of MCP tool objects (with .name, .description, .inputSchema attributes or dicts)
        Returns:
            List of dicts in OpenAI tool schema format.
        """
        openai_tools = []
        for tool in mcp_tools:
            # Handle both dict and object style
            if isinstance(tool, dict):
                name = tool.get("name")
                description = tool.get("description")
                parameters = tool.get("inputSchema") or tool.get("parameters")
            else:
                name = getattr(tool, "name", None)
                description = getattr(tool, "description", None)
                parameters = getattr(tool, "inputSchema", None)
            if parameters is None:
                # Fallback to empty object
                parameters = {"type": "object", "properties": {}, "required": []}
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                },
            })
        return openai_tools
