#!/usr/bin/env python3
__import__('sys').path.append('..')
from client import Mosaic

# Server Configurations

SERVER_CONFIGS = [
    {
        "name": "database_server",
        "description": "A server that manages a SQLite database with full CRUD operations.",
        "url": "http://localhost:8000/sse"
    },
]

# Main Function

def main():
    """Main function to run the Mosaic client."""
    mosaic = Mosaic.create(
        server_configs=SERVER_CONFIGS,
        web_search=True,
    )
    mosaic.run()

# Main Execution

if __name__ == "__main__":
    main()
