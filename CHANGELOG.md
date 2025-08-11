# Changelog

All notable changes to the Mosaic project will be documented in this file.


## [1.2.0] - 2025-08-11

### Changed
- **Client**: Cleaned up client.py and added proper documentation with comments
- **New Model**: Updated code to use the GPT-5 model released by OpenAI

### Fixed
- **Tavily**: Fixed Tavily-Langchain depriciation error warning, to use new integrated library

## [1.1.0] - 2025-06-27

### Added
- **Multi-Agent Client Framework**: Core framework for connecting to MCP servers
- **Built-in Capabilities**: Web search, RAG, and general conversation agents
- **MCP Server Integration**: Connect to any MCP-compatible server
- **Configuration Templates**: Ready-to-use server configuration templates
- **Comprehensive Documentation**: Complete README.md with detailed setup instructions and examples
- **Example MCP Servers**: Database and calendar servers as add-ons
- **Enhanced Error Handling**: Better user feedback and troubleshooting information
- **Professional Project Structure**: Clean organization for GitHub distribution

### Changed
- **Enhanced README.md**: Comprehensive documentation with clear sections, and step-by-step guides
- **Improved requirements.txt**: Better organization with categorized dependencies and system requirements
- **Better Code Documentation**: Enhanced comments and docstrings throughout the codebase
- **Simplified Configuration**: Streamlined server configuration and setup process
- **User Experience**: More intuitive usage with direct client.py and template options
- **Architecture Focus**: Positioned as MCP client framework rather than toolkit

### Fixed
- **Documentation Issues**: Clearer setup instructions and troubleshooting guides
- **Configuration Clarity**: Better examples and templates for server setup
- **Installation Process**: Simplified dependency management and environment setup

## [1.0.0] - 2025-06-23

### Added
- Initial release of Mosaic multi-agent client framework
- Modular multi-agent architecture with intelligent query routing
- Built-in web search agent using Tavily API for real-time information
- Built-in RAG agent for document analysis and processing
- Built-in general conversation agent for context-aware responses
- MCP (Model Context Protocol) server integration
- PDF and image processing capabilities
- Vector search using FAISS
- Conversation history management
- Configurable server architecture
- Support for custom MCP servers
- Example database server as add-on

### Features
- **Multi-Agent Client Framework**: Intelligent routing of queries to specialized agents and MCP servers
- **Built-in Capabilities**: Web search, RAG, and general conversation
- **MCP Server Integration**: Connect to any MCP-compatible server
- **OpenAI + LangChain**: Leverage GPT models with custom tools
- **Conversation Management**: Context-aware conversations across agents
- **Modular Design**: Easy to extend with custom agents and MCP servers
- **Configuration Flexibility**: Customizable server configurations and settings

### Technical Details
- Built with LangChain and LangGraph for AI orchestration
- Uses OpenAI GPT models for natural language processing
- Implements MCP for server communication
- Supports async/await patterns for better performance
- Includes comprehensive logging and error handling
- Modular codebase for easy customization and extension

### Dependencies
- Python 3.8+
- LangChain ecosystem
- OpenAI API
- Tavily API
- FAISS for vector search
- Various document processing libraries

---

## Version History Summary

### Version 1.1.0 (Current) - 2025-06-27
- **Multi-Agent Client Framework**: Core framework for MCP server integration
- **Built-in Capabilities**: Web search, RAG, and general conversation
- **Example Servers**: Database and calendar servers as add-ons
- **Professional Setup**: Clean project structure ready for GitHub

### Version 1.0.0 - 2025-06-23
- **Initial Release**: Multi-agent client framework with built-in capabilities
- **Core Features**: Modular architecture, intelligent routing, conversation management
- **Integration**: MCP servers, OpenAI GPT models, Tavily web search
- **Add-ons**: Example database server for demonstration

---

## Contributing

When contributing to this project, please update this changelog with your changes following the format above.

## Release Process

1. Update version numbers in relevant files
2. Update this CHANGELOG.md with new version
3. Create a git tag for the release
4. Update README.md if needed
5. Test the release thoroughly
6. Publish to GitHub

---

*For more information about releases, see the [GitHub releases page](https://github.com/garvit-mehra/mosaic/releases).* 
