#!/usr/bin/env python3
"""
Mosaic Usage Template
=====================

This template demonstrates how to use the modular Mosaic client with your own server configurations.
You can add/remove servers, and toggle web search or RAG, all from this file.

Author: Garvit Mehra
Version: 1.1.0
License: MIT

Usage:
    python mosaic_template.py
"""

__import__('sys').path.append('..')
from client import Mosaic

# =============================================================================
# Server Configurations
# =============================================================================

SERVER_CONFIGS = [
    {
        "name": "database_server",
        "description": "A server that manages a SQLite database with full CRUD operations.",
        "url": "http://localhost:8000/sse"
    },
]

# =============================================================================
# Main Function
# =============================================================================

def main():
    """Main function to run the Mosaic client."""
    mosaic = Mosaic.create(
        server_configs=SERVER_CONFIGS,
        web_search=True,
        rag=True
    )
    mosaic.run()

# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    main()
