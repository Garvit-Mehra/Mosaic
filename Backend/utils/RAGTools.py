import contextvars
from langchain.tools import tool
from .ProcessPDF import load_document_tool, query_documents_tool, list_documents_tool, clear_documents_tool

# Context variable to hold the active conversation ID during agent execution
conversation_context = contextvars.ContextVar('conversation_context', default="temp")

@tool
def load_document(file_path: str) -> str:
    """
    Load and process a document file (PDF or image) into the knowledge base.
    
    Args:
        file_path: Path to the document file (PDF, JPG, PNG, etc.)
    
    Returns:
        Status message indicating success or failure
    """
    conv_id = conversation_context.get()
    return load_document_tool(file_path, conv_id)


@tool
def query_documents(query: str) -> str:
    """
    Search and retrieve relevant information from loaded documents using RAG.
    
    Args:
        query: The search query to find relevant information in documents
    
    Returns:
        Relevant information from documents with relevance scores
    """
    conv_id = conversation_context.get()
    return query_documents_tool(query, conv_id)


@tool
def list_documents() -> str:
    """
    List all currently loaded documents in the knowledge base.
    
    Returns:
        Summary of all loaded documents
    """
    conv_id = conversation_context.get()
    return list_documents_tool(conv_id)


@tool
def clear_documents() -> str:
    """
    Clear all loaded documents from the knowledge base.
    
    Returns:
        Confirmation message
    """
    conv_id = conversation_context.get()
    return clear_documents_tool(conv_id) 