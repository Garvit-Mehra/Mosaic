"""
Mosaic Job Handlers

Actual execution logic for background jobs. Each handler is called by
TaskQueue workers when a job of the corresponding type is dequeued.
"""

import os
import sys
import logging
import asyncio
import aiohttp
from typing import Any, Dict

# Ensure the Backend directory is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

logger = logging.getLogger(__name__)


# =============================================================================
# Handler: MCP Tool Call
# =============================================================================

async def handle_mcp_tool_call(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute an MCP tool call in the background.

    Payload:
        server_url: MCP server URL
        server_name: name for the server
        tool_name: tool to invoke
        args: dict of arguments
    """
    server_url = payload.get("server_url")
    server_name = payload.get("server_name", "mcp_server")
    tool_name = payload.get("tool_name")
    args = payload.get("args", {})

    if not server_url or not tool_name:
        return {"status": "failed", "error": "server_url and tool_name are required"}

    logger.info(f"MCP tool call: {server_name}/{tool_name}")

    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient

        client = MultiServerMCPClient({
            server_name: {
                "url": server_url,
                "transport": "streamable_http",
            }
        })

        # Get tools and find the one we need
        tools = await client.get_tools(server_name=server_name)
        target_tool = next((t for t in tools if t.name == tool_name), None)

        if not target_tool:
            # Try SSE fallback
            client = MultiServerMCPClient({
                server_name: {
                    "url": server_url,
                    "transport": "sse",
                }
            })
            tools = await client.get_tools(server_name=server_name)
            target_tool = next((t for t in tools if t.name == tool_name), None)

        if not target_tool:
            return {"status": "failed", "error": f"Tool '{tool_name}' not found on server"}

        # Invoke the tool
        result = await target_tool.ainvoke(args)

        return {
            "status": "completed",
            "server": server_name,
            "tool": tool_name,
            "result": result if isinstance(result, (str, dict, list)) else str(result),
        }

    except Exception as e:
        logger.error(f"MCP tool call failed: {e}")
        return {"status": "failed", "error": str(e)}


# =============================================================================
# Handler: RAG Document Processing
# =============================================================================

async def handle_rag_process(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a document for RAG indexing.

    Payload:
        file_path: path to the document (PDF or image)
        chunk_size: optional, default 1000
        chunk_overlap: optional, default 200
    """
    file_path = payload.get("file_path")
    chunk_size = payload.get("chunk_size", 1000)
    chunk_overlap = payload.get("chunk_overlap", 200)

    if not file_path:
        return {"status": "failed", "error": "file_path is required"}

    if not os.path.exists(file_path):
        return {"status": "failed", "error": f"File not found: {file_path}"}

    logger.info(f"RAG processing: {file_path}")

    try:
        from utils.ProcessPDF import process_file, add_document_to_store

        # Process the file (extract text, tables, images)
        content = process_file(file_path)

        if not content or not content.strip():
            return {"status": "failed", "error": "No content could be extracted from the file"}

        # Index into vector store
        result_msg = add_document_to_store(file_path, content)

        return {
            "status": "completed",
            "file_path": file_path,
            "content_length": len(content),
            "message": result_msg,
        }

    except FileNotFoundError as e:
        return {"status": "failed", "error": str(e)}
    except Exception as e:
        logger.error(f"RAG processing failed: {e}")
        return {"status": "failed", "error": str(e)}


# =============================================================================
# Handler: Web Scrape
# =============================================================================

async def handle_web_scrape(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fetch content from a URL and return extracted text.

    Payload:
        url: URL to fetch
        max_length: max characters to return (default 10000)
        extract: "text" (default) or "html"
    """
    url = payload.get("url")
    max_length = payload.get("max_length", 10000)
    extract = payload.get("extract", "text")

    if not url:
        return {"status": "failed", "error": "url is required"}

    if not url.startswith("http://") and not url.startswith("https://"):
        return {"status": "failed", "error": "URL must start with http:// or https://"}

    logger.info(f"Web scrape: {url}")

    try:
        connector = aiohttp.TCPConnector(ssl=False)
        timeout = aiohttp.ClientTimeout(total=30)

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            async with session.get(url) as response:
                if response.status >= 400:
                    return {
                        "status": "failed",
                        "error": f"HTTP {response.status} from {url}",
                    }

                content_type = response.headers.get("Content-Type", "")
                raw_content = await response.text()

                if extract == "html":
                    content = raw_content[:max_length]
                else:
                    # Strip HTML tags for plain text extraction
                    import re
                    # Remove script and style elements
                    text = re.sub(r'<script[^>]*>.*?</script>', '', raw_content, flags=re.DOTALL)
                    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
                    # Remove HTML tags
                    text = re.sub(r'<[^>]+>', ' ', text)
                    # Clean whitespace
                    text = re.sub(r'\s+', ' ', text).strip()
                    content = text[:max_length]

                return {
                    "status": "completed",
                    "url": url,
                    "content_type": content_type,
                    "content_length": len(content),
                    "content": content,
                }

    except asyncio.TimeoutError:
        return {"status": "failed", "error": f"Timeout fetching {url}"}
    except Exception as e:
        logger.error(f"Web scrape failed: {e}")
        return {"status": "failed", "error": str(e)}


# =============================================================================
# Handler: Batch Chat
# =============================================================================

async def handle_batch_chat(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process multiple chat messages sequentially through the Mosaic agent.

    Payload:
        messages: list of message strings
        conversation_id: conversation to append to (optional)
        user_id: user context
    """
    messages = payload.get("messages", [])
    conversation_id = payload.get("conversation_id")
    user_id = payload.get("user_id", "system")

    if not messages:
        return {"status": "failed", "error": "messages list is empty"}

    logger.info(f"Batch chat: {len(messages)} messages, user={user_id}")

    try:
        from client import AgentRegistry, MosaicHandler
        from utils.ConversationDB import ConversationManager

        # Create a handler instance
        conversation_db = ConversationManager()
        registry = AgentRegistry()

        # Initialize if not already done (workers are separate processes)
        if not registry._initialized:
            from dotenv import load_dotenv
            load_dotenv()
            server_configs = []  # Workers don't need MCP servers
            await registry.initialize(server_configs, web_search=False)

        handler = MosaicHandler(registry, conversation_db)

        results = []
        for i, message in enumerate(messages):
            try:
                result = await handler.chat(
                    message=message,
                    conversation_id=conversation_id,
                    user_id=user_id,
                )
                results.append({
                    "index": i,
                    "message": message[:50],
                    "response": result.get("response", "")[:200],
                    "agent": result.get("agent"),
                })

                # Persist to conversation
                if conversation_id:
                    conversation_db.add_message(conversation_id, "user", message)
                    conversation_db.add_message(
                        conversation_id, "assistant", result["response"],
                        agent=result.get("agent")
                    )

            except Exception as e:
                results.append({
                    "index": i,
                    "message": message[:50],
                    "error": str(e),
                })

        return {
            "status": "completed",
            "processed": len(results),
            "successful": len([r for r in results if "error" not in r]),
            "failed": len([r for r in results if "error" in r]),
            "results": results,
        }

    except Exception as e:
        logger.error(f"Batch chat failed: {e}")
        return {"status": "failed", "error": str(e)}


# =============================================================================
# Handler Registry
# =============================================================================

MOSAIC_HANDLERS = {
    "mcp_tool_call": handle_mcp_tool_call,
    "rag_process": handle_rag_process,
    "web_scrape": handle_web_scrape,
    "batch_chat": handle_batch_chat,
}
