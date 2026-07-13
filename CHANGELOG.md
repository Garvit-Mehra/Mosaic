# Changelog

All notable changes to the Mosaic project will be documented in this file.

## [2.0.0] - 2025-07-13

### Added
- **Full-Stack Application**: Next.js 15 frontend with real-time streaming chat UI
- **Streaming Responses**: Token-by-token SSE streaming from backend to frontend
- **Conversation Persistence**: SQLite-backed conversation history via SQLAlchemy
- **Settings UI**: Add/remove/monitor MCP servers from the browser (`/settings`)
- **Hot-Reload MCP Servers**: `POST /servers/refresh` picks up newly started servers without restart
- **Dynamic Server Management**: Add MCP servers at runtime via API or UI
- **Server Tools Viewer**: See all tools a connected MCP server provides
- **Conversation CRUD**: Full REST API for conversations (create, list, get, update, delete)
- **Auto-Conversation Creation**: First message auto-creates a conversation if none specified
- **Collapsible Sidebar**: Shows real conversations from the database with delete support

### Changed
- **Switched to Ollama**: Removed OpenAI dependency — all inference runs locally via Ollama
- **Improved Agent Routing**: Tighter classification prompt so general queries don't misroute to RAG
- **Fixed FastAPI Lifespan**: App now properly initializes on startup via `lifespan`
- **Updated LangChain Imports**: Fixed deprecated `langchain.text_splitter` and `langchain.schema` paths
- **Agent Descriptions**: Made descriptions explicit to prevent routing confusion
- **MCP Servers Optional**: Backend starts fine with zero MCP servers running

### Removed
- **Dead Code**: Removed `utils/Clients.py` (300+ lines of unused OpenAI/Autogen clients)
- **Broken Streaming Endpoint**: Removed old `/chat/stream` that referenced non-existent methods
- **Auth Dependency**: Removed NextAuth/Prisma from frontend (simplifies setup)
- **Hardcoded Sidebar**: Replaced static chat list with live data from backend

### Fixed
- **Message Model**: `ChatRequest` now correctly includes `conversation_id` and `user_id`
- **Session Management**: ConversationDB uses proper context manager with commit/rollback
- **Import Chain**: Fixed `langchain.text_splitter` → `langchain_text_splitters`
- **Import Chain**: Fixed `langchain.schema.Document` → `langchain_core.documents.Document`
- **CSS Variables**: Added missing `--foreground`, `--background`, `--input-bg` definitions
- **Frontend .env**: Fixed backend URL pointing to wrong port (was 8000, now 8080)

---

## [1.2.0] - 2025-08-11

### Changed
- **Client**: Cleaned up client.py and added proper documentation with comments
- **New Model**: Updated code to use the GPT-5 model released by OpenAI

### Fixed
- **Tavily**: Fixed Tavily-Langchain depreciation error warning, to use new integrated library

---

## [1.1.0] - 2025-06-27

### Added
- **Multi-Agent Client Framework**: Core framework for connecting to MCP servers
- **Built-in Capabilities**: Web search, RAG, and general conversation agents
- **MCP Server Integration**: Connect to any MCP-compatible server
- **Configuration Templates**: Ready-to-use server configuration templates
- **Example MCP Servers**: Database and calendar servers as add-ons

### Changed
- **Enhanced README.md**: Comprehensive documentation
- **Improved requirements.txt**: Categorized dependencies
- **Architecture Focus**: Positioned as MCP client framework

---

## [1.0.0] - 2025-06-23

### Added
- Initial release of Mosaic multi-agent client framework
- Modular multi-agent architecture with intelligent query routing
- PDF and image processing capabilities
- Vector search using FAISS
- Conversation history management (in-memory)
- Support for custom MCP servers
- Example database server
