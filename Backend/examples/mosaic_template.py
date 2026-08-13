#!/usr/bin/env python3
import asyncio
import os
import sys

# Ensure the Backend directory is importable if running from examples
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client import AgentRegistry, MosaicHandler
from utils.ConversationDB import ConversationManager

# Example server configurations
SERVER_CONFIGS = [
    {
        "name": "example_server",
        "description": "An example MCP server.",
        "url": "http://localhost:8000/sse",
        "transport": "sse"
    },
]

async def main():
    print("Initializing Mosaic...")
    
    # 1. Initialize databases
    conversation_db = ConversationManager()
    
    # 2. Initialize Agent Registry
    registry = AgentRegistry()
    # Optionally load MCP servers and enable/disable web search
    await registry.initialize(SERVER_CONFIGS, web_search=False)
    
    # 3. Create Handler
    handler = MosaicHandler(registry, conversation_db)
    
    print("\nMosaic initialized. Type 'quit' to exit.")
    conversation_id = None
    
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ['quit', 'exit']:
                break
                
            print("\nMosaic: ", end="", flush=True)
            
            # Use streaming chat
            async for chunk in handler.chat_stream(
                user_input, 
                conversation_id=conversation_id,
                user_id="template_user"
            ):
                if chunk["type"] == "token":
                    print(chunk["content"], end="", flush=True)
                elif chunk["type"] == "done":
                    conversation_id = chunk.get("conversation_id", conversation_id)
                    print(f"\n[Agent used: {chunk.get('agent', 'unknown')}]")
                    
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    asyncio.run(main())