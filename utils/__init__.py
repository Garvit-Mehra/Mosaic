"""
Mosaic - Modular Multi-Agent Tools for Python

This package contains utility modules for the Mosaic toolkit:
- RAGTools: Document processing and retrieval capabilities
- ProcessPDF: PDF and image processing functionality
- Clients: MCP client implementations (legacy)

Author: Garvit Mehra
Version: 1.0.0
"""

from .RAGTools import load_document, query_documents, list_documents, clear_documents
from .ProcessPDF import process_file, search_documents, get_document_summary

__version__ = "1.0.0"
__author__ = "Garvit Mehra"

# Export main functions for easy access
__all__ = [
    # RAG Tools
    "load_document",
    "query_documents", 
    "list_documents",
    "clear_documents",
    
    # PDF Processing
    "process_file",
    "search_documents",
    "get_document_summary",
] 