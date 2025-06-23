import os
import io
import camelot
import PyPDF2
from PIL import Image
from dotenv import load_dotenv
from pdf2image import convert_from_path
import openai
import base64
import json
from typing import List, Dict, Any
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Try multiple possible .env file locations
load_dotenv("../.env")
load_dotenv(".env")
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

# Global storage for processed documents
document_store = {}
vector_store = None
_embeddings = None  # Lazy initialization

def get_embeddings():
    """Get embeddings instance with lazy initialization"""
    global _embeddings
    if _embeddings is None:
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not set in environment variables.")
        _embeddings = OpenAIEmbeddings()
    return _embeddings

def extract_text_from_pdf(pdf_path):
    text_content = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                text_content += f"\n\n--- Page {i + 1} ---\n{text}"
            else:
                text_content += f"\n\n--- Page {i + 1} ---\n[No extractable text]"
    return text_content


def extract_tables_from_pdf(pdf_path):
    tables_str = ""
    try:
        tables = camelot.read_pdf(pdf_path, pages="all", flavor="lattice")
        for idx, table in enumerate(tables):
            tables_str += f"\n\n--- Table {idx + 1} ---\n{table.df.to_string(index=False, header=True)}"
    except Exception as e:
        tables_str += f"\n\n[Table extraction failed: {str(e)}]"
    return tables_str


def image_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def describe_image(image: Image.Image) -> str:
    try:
        image_b64 = image_to_base64(image)
        response = openai.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image in detail, including any text, charts, diagrams, or visual elements you can see."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
                    ]
                }
            ],
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Image description failed: {str(e)}]"


def extract_and_describe_images(pdf_path):
    image_descriptions = ""
    try:
        pages = convert_from_path(pdf_path)
        for i, image in enumerate(pages):
            desc = describe_image(image)
            image_descriptions += f"\n\n--- Image of Page {i + 1} ---\n{desc}"
    except Exception as e:
        image_descriptions += f"\n\n[Image extraction failed: {str(e)}]"
    return image_descriptions


def process_image(image_path: str) -> str:
    """Process a single image file and return its description"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")
    
    try:
        image = Image.open(image_path)
        description = describe_image(image)
        return f"--- Image: {os.path.basename(image_path)} ---\n{description}"
    except Exception as e:
        return f"[Image processing failed: {str(e)}]"


def process_pdf(pdf_path: str) -> str:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File not found: {pdf_path}")

    text = extract_text_from_pdf(pdf_path)
    tables = extract_tables_from_pdf(pdf_path)
    images = extract_and_describe_images(pdf_path)

    return f"{text}\n\n{tables}\n\n{images}".strip()


def process_file(file_path: str) -> str:
    """Process any supported file type (PDF or image)"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.pdf':
        return process_pdf(file_path)
    elif file_ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']:
        return process_image(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_ext}")


def create_document_chunks(content: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """Split document content into chunks for vector storage"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    
    chunks = text_splitter.split_text(content)
    documents = [Document(page_content=chunk, metadata={"source": "document"}) for chunk in chunks]
    return documents


def add_document_to_store(file_path: str, content: str) -> str:
    """Add a processed document to the vector store"""
    global vector_store, document_store
    
    # Store the full content
    document_store[file_path] = content
    
    # Create chunks and add to vector store
    chunks = create_document_chunks(content)
    
    if vector_store is None:
        vector_store = FAISS.from_documents(chunks, get_embeddings())
    else:
        vector_store.add_documents(chunks)
    
    return f"Document '{os.path.basename(file_path)}' added to knowledge base with {len(chunks)} chunks."


def search_documents(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """Search documents using vector similarity"""
    global vector_store
    
    if vector_store is None:
        return [{"content": "No documents have been loaded yet.", "score": 0.0}]
    
    try:
        results = vector_store.similarity_search_with_score(query, k=k)
        return [
            {
                "content": doc.page_content,
                "score": float(score),
                "metadata": doc.metadata
            }
            for doc, score in results
        ]
    except Exception as e:
        return [{"content": f"Search failed: {str(e)}", "score": 0.0}]


def get_document_summary(file_path: str) -> str:
    """Get a summary of what documents are available"""
    global document_store
    
    if not document_store:
        return "No documents have been loaded yet."
    
    summary = "Available documents:\n"
    for path, content in document_store.items():
        filename = os.path.basename(path)
        content_length = len(content)
        summary += f"- {filename} ({content_length} characters)\n"
    
    return summary


def clear_documents() -> str:
    """Clear all loaded documents"""
    global document_store, vector_store
    
    document_store.clear()
    vector_store = None
    return "All documents cleared from memory."


# Tool functions for the RAG agent
def load_document_tool(file_path: str) -> str:
    """Load and process a document file (PDF or image)"""
    try:
        content = process_file(file_path)
        result = add_document_to_store(file_path, content)
        return f"Successfully loaded document: {result}"
    except Exception as e:
        return f"Error loading document: {str(e)}"


def query_documents_tool(query: str) -> str:
    """Query the loaded documents using RAG"""
    try:
        results = search_documents(query, k=3)
        
        if not results or results[0]["content"] == "No documents have been loaded yet.":
            return "No documents are loaded. Please load a document first using the load_document tool."
        
        response = "Relevant information from documents:\n\n"
        for i, result in enumerate(results, 1):
            response += f"--- Result {i} (Relevance: {result['score']:.3f}) ---\n"
            response += f"{result['content']}\n\n"
        
        return response
    except Exception as e:
        return f"Error querying documents: {str(e)}"


def list_documents_tool() -> str:
    """List all loaded documents"""
    return get_document_summary()


def clear_documents_tool() -> str:
    """Clear all loaded documents"""
    return clear_documents()
