from typing import List
from langchain.tools import tool
from .ProcessPDF import load_document_tool, query_documents_tool, list_documents_tool, clear_documents_tool


@tool
def load_document(file_path: str) -> str:
    """
    Load and process a document file (PDF or image) into the knowledge base.
    
    Args:
        file_path: Path to the document file (PDF, JPG, PNG, etc.)
    
    Returns:
        Status message indicating success or failure
    """
    return load_document_tool(file_path)


@tool
def query_documents(query: str) -> str:
    """
    Search and retrieve relevant information from loaded documents using RAG.
    
    Args:
        query: The search query to find relevant information in documents
    
    Returns:
        Relevant information from documents with relevance scores
    """
    return query_documents_tool(query)


@tool
def list_documents() -> str:
    """
    List all currently loaded documents in the knowledge base.
    
    Returns:
        Summary of all loaded documents
    """
    return list_documents_tool()


@tool
def clear_documents() -> str:
    """
    Clear all loaded documents from the knowledge base.
    
    Returns:
        Confirmation message
    """
    return clear_documents_tool() 