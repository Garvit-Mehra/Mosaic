# Mosaic - Multi-Agent Client for MCP Servers

A modern multi-agent client framework that connects to MCP (Model Context Protocol) servers using LangChain and local LLMs.

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

---

## Key Features

- Multi-agent client with intelligent query routing  
- Web search and document analysis (RAG) built-in  
- Connects to any MCP-compatible server  
- Works with local LLMs via [Ollama](https://ollama.com)  
- Easy to extend with new agents and tools  

---

## System Requirements for Running Mistral 7B via Ollama

To run Mistral 7B locally using Ollama, here are the essential hardware and setup specifications:

| Component       | Minimum Requirements                          | Recommended Setup (Smoother Performance)           |
|-----------------|-----------------------------------------------|----------------------------------------------------|
| **System RAM**  | 16 GB (for CPU-only or 4/5-bit quantized runs) | 32 GB+ for more headroom and multitasking |
| **GPU (VRAM)**  | 6 GB+ VRAM may work; 12 GB (e.g., RTX 3060) recommended | RTX 3090 (24 GB) or better for faster and smoother runs |
| **CPU**         | Mid-range multi-core CPU (e.g., i7 8th gen or Ryzen 5 3rd gen) | Higher core counts (i9 / Ryzen 7+) for heavier workloads |
| **Storage**     | 10-15 GB (for Ollama, model download) | 15 GB+ if you manage multiple models or data |
| **Software**    | Ollama installed, OS: Windows 10+ / macOS / Ubuntu 20.04+ | Same, with optional CUDA setup on GPU machines |

### Notes
- **Quantization** (4-bit or 5-bit) reduces both VRAM and RAM usage significantly.  
- **CPU-only setups** are possible but slower; high-end CPUs with 64 GB RAM recommended.  
- **Ollama** simplifies deployment; model size for Mistral 7B typically 4–8 GB.  

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
4. **Download and run Mistral via Ollama**
    ```bash
    ollama pull mistral
    ollama run mistral
    ```
5. **Run Mosaic**
    ```bash
    cd examples
    python mosaic_template.py
    ```

---

## Configuration

**MCP Server Example**
```python
{
    "name": "your_server_name",
    "description": "Description of what your server does.",
    "url": "http://localhost:PORT/sse",
    "transport": "sse"
}
```

**Client Configuration**
```python
MODEL_NAME = "mistral"   # Change AI model
MAX_HISTORY_EXCHANGES = 5
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
- Community contributors

---
