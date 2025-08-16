# Mosaic - Multi-Agent Client for MCP Servers

A modern multi-agent client framework that connects to MCP (Model Context Protocol) servers using LangChain and locally running LLMs (via [Ollama](https://ollama.com)).

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/Version-1.3.0-orange.svg)](VERSION)
[![LangChain](https://img.shields.io/badge/LangChain-0.1+-yellow.svg)](https://langchain.com)
[![MCP](https://img.shields.io/badge/MCP-1.0+-purple.svg)](https://modelcontextprotocol.io/)

---

## Overview

**Mosaic** is a modular multi-agent client framework that intelligently routes user queries to specialized agents and MCP (Model Context Protocol) servers.  

It supports:  
- Multi-agent orchestration  
- Web search integration  
- Retrieval-Augmented Generation (RAG)  
- MCP server connectivity  
- Extensible agent framework  
- Local LLMs via Ollama (choose any available model, e.g., Llama 3, Mistral, Gemma, etc.)  

---

## Key Features

- Multi-agent client with intelligent query routing  
- Built-in web search and document analysis (RAG)  
- Works with any model available through Ollama  
- Easy extension with new agents and MCP servers  

---

## Installation & Setup

### Prerequisites
- Python 3.8+  
- Ollama installed ([Download](https://ollama.com))  
- Tavily API key (for web search)  

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/garvit-mehra/mosaic.git
   cd mosaic
   ```

2. **Create a virtual environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```
4. **Download and run the required model via Ollama**
    ```bash
    ollama run [any model of your choice]
    ```
5. **Run Mosaic**
    ```bash
    cd examples
    python mosaic_template.py
    ```

---

## Configuration

**Model configuration**

To change the model you are using, update ```model_config``` within ```mosaic_template.py```

**MCP Server Example**
```python
{
    "name": "your_server_name",
    "description": "Description of what your server does.",
    "url": "http://localhost:PORT/sse",
    "transport": "sse"
}
```

---

## License

This project is distributed under a **Non-Commercial, No-Distribution License (Based on MIT)**.

- **No Commercial Use**: The Software may not be used, in whole or in part, for any commercial purpose.  
- **No Distribution**: Redistribution, sublicensing, or selling of the Software, in original or modified form, is prohibited.  
- **No Hosting as a Service**: The Software may not be offered as part of a hosted or managed service, whether free or paid.  

The full license text can be found in the [LICENSE](LICENSE) file.

---

## Acknowledgments
- LangChain for the AI framework
- MCP for the server protocol
- Tavily for web search integration
- Ollama for local LLM deployment
- Community contributors

---
