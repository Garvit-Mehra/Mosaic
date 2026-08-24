import os
import io
import camelot
import warnings
import json

# Suppress known harmless warnings from external libraries
warnings.filterwarnings("ignore", message="The NumPy module was reloaded")

import pypdf
from PIL import Image
from dotenv import load_dotenv
from pdf2image import convert_from_path
import openai
import base64
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma

# Try multiple possible .env file locations
load_dotenv("../.env")
load_dotenv(".env")
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
METADATA_FILE = os.path.join(CHROMA_PERSIST_DIR, "doc_metadata.json")

_embeddings = None
_vector_store = None
_document_metadata = {} # Format: { "conversation_id": { "filename": { "size_chars": 123 } } }

def load_document_metadata():
    global _document_metadata
    if os.path.exists(METADATA_FILE):
        try:
            with open(METADATA_FILE, "r") as f:
                _document_metadata = json.load(f)
        except Exception:
            _document_metadata = {}

def save_document_metadata():
    global _document_metadata
    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
    with open(METADATA_FILE, "w") as f:
        json.dump(_document_metadata, f)

load_document_metadata()

def get_embeddings():
    """Get embeddings instance based on LLM_PROVIDER"""
    global _embeddings
    if _embeddings is None:
        provider = os.getenv("LLM_PROVIDER", "ollama").lower()
        
        if provider == "ollama":
            from langchain_ollama import OllamaEmbeddings
            base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434")
            _embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=base_url)
        else:
            if not os.getenv("OPENAI_API_KEY") and not os.getenv("LLM_API_KEY"):
                raise ValueError(f"{provider} requires an API key for embeddings.")
            
            from langchain_openai import OpenAIEmbeddings
            kwargs = {}
            if provider == "compatible" and os.getenv("LLM_BASE_URL"):
                kwargs["openai_api_base"] = os.getenv("LLM_BASE_URL")
                kwargs["openai_api_key"] = os.getenv("LLM_API_KEY", "not-needed")
            else:
                kwargs["openai_api_key"] = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
                
            _embeddings = OpenAIEmbeddings(**kwargs)
            
    return _embeddings

def get_vector_store():
    """Get or initialize the persistent Chroma DB"""
    global _vector_store
    if _vector_store is None:
        _vector_store = Chroma(
            embedding_function=get_embeddings(),
            persist_directory=CHROMA_PERSIST_DIR
        )
    return _vector_store

def extract_text_from_pdf(pdf_path):
    text_content = ""
    with open(pdf_path, "rb") as file:
        reader = pypdf.PdfReader(file)
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
    if not openai.api_key:
        return "[Image description skipped: OPENAI_API_KEY not set]"
        
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
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.pdf':
        return process_pdf(file_path)
    elif file_ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']:
        return process_image(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_ext}")


def create_document_chunks(content: str, filename: str, conversation_id: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    
    chunks = text_splitter.split_text(content)
    documents = [Document(page_content=chunk, metadata={"source": filename, "conversation_id": conversation_id}) for chunk in chunks]
    return documents


def add_document_to_store(file_path: str, content: str, conversation_id: str, original_filename: str = None) -> str:
    global _document_metadata
    
    filename = original_filename or os.path.basename(file_path)
    conv_id = str(conversation_id)
    
    if conv_id not in _document_metadata:
        _document_metadata[conv_id] = {}
        
    _document_metadata[conv_id][filename] = {
        "size_chars": len(content),
    }
    save_document_metadata()
    
    chunks = create_document_chunks(content, filename, conv_id)
    vs = get_vector_store()
    vs.add_documents(chunks)
    
    return f"Document '{filename}' added to knowledge base with {len(chunks)} chunks."


def search_documents(query: str, conversation_id: str, k: int = 5) -> List[Dict[str, Any]]:
    vs = get_vector_store()
    conv_id = str(conversation_id)
    
    try:
        # Filter by conversation_id to strictly isolate documents
        results = vs.similarity_search_with_score(query, k=k, filter={"conversation_id": conv_id})
        if not results:
             return [{"content": "No documents found matching the query.", "score": 0.0}]
             
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


def get_document_summary(conversation_id: str) -> str:
    global _document_metadata
    conv_id = str(conversation_id)
    
    if conv_id not in _document_metadata or not _document_metadata[conv_id]:
        return "No documents have been loaded yet."
    
    summary = "Available documents in knowledge base:\n"
    for filename, meta in _document_metadata[conv_id].items():
        summary += f"- {filename} ({meta.get('size_chars', 0)} characters)\n"
    
    return summary
    
def get_documents_list(conversation_id: str) -> List[Dict[str, Any]]:
    global _document_metadata
    conv_id = str(conversation_id)
    if conv_id not in _document_metadata:
        return []
        
    return [{"filename": fname, "size_chars": meta.get("size_chars", 0)} for fname, meta in _document_metadata[conv_id].items()]


def remove_document(filename: str, conversation_id: str) -> str:
    global _document_metadata
    conv_id = str(conversation_id)
    
    if conv_id in _document_metadata and filename in _document_metadata[conv_id]:
        del _document_metadata[conv_id][filename]
        save_document_metadata()
        
    vs = get_vector_store()
    try:
        vs._collection.delete(where={"$and": [{"source": filename}, {"conversation_id": conv_id}]})
    except Exception as e:
        pass 
        
    return f"Removed {filename}"
    

def clear_documents(conversation_id: str) -> str:
    global _document_metadata
    conv_id = str(conversation_id)
    
    if conv_id in _document_metadata:
        del _document_metadata[conv_id]
        save_document_metadata()
        
    vs = get_vector_store()
    try:
        vs._collection.delete(where={"conversation_id": conv_id})
    except Exception:
        pass
    
    return f"All documents cleared from conversation {conv_id}."
    

def migrate_documents(old_id: str, new_id: str) -> str:
    global _document_metadata
    old_id = str(old_id)
    new_id = str(new_id)
    
    if old_id not in _document_metadata or not _document_metadata[old_id]:
        return "No documents to migrate"
        
    if new_id not in _document_metadata:
        _document_metadata[new_id] = {}
        
    # Migrate in metadata dict
    for fname, meta in _document_metadata[old_id].items():
        _document_metadata[new_id][fname] = meta
        
    del _document_metadata[old_id]
    save_document_metadata()
    
    # Migrate in Chroma
    vs = get_vector_store()
    try:
        # Get all ids matching old_id
        res = vs._collection.get(where={"conversation_id": old_id})
        if res and res['ids']:
            new_metadatas = []
            for meta in res['metadatas']:
                meta['conversation_id'] = new_id
                new_metadatas.append(meta)
                
            vs._collection.update(ids=res['ids'], metadatas=new_metadatas)
    except Exception as e:
        print(f"Error migrating Chroma vectors: {e}")
        
    return f"Migrated documents to conversation {new_id}"


# -------------------------------------------------------------------------
# Tool functions for the RAG agent
# NOTE: We can no longer just use generic tools without conversation context.
# We will rely on LangChain's injected tool arguments or a context var.
# -------------------------------------------------------------------------

def load_document_tool(file_path: str, conversation_id: str) -> str:
    """Agent tool to load a local document file"""
    try:
        content = process_file(file_path)
        result = add_document_to_store(file_path, content, conversation_id)
        return f"Successfully loaded document: {result}"
    except Exception as e:
        return f"Error loading document: {str(e)}"


def query_documents_tool(query: str, conversation_id: str) -> str:
    """Agent tool to query the knowledge base"""
    try:
        results = search_documents(query, conversation_id, k=3)
        
        response = "Relevant information from documents:\n\n"
        for i, result in enumerate(results, 1):
            source = result.get('metadata', {}).get('source', 'Unknown')
            response += f"--- Result {i} (Source: {source}, Distance: {result['score']:.3f}) ---\n"
            response += f"{result['content']}\n\n"
        
        return response
    except Exception as e:
        return f"Error querying documents: {str(e)}"


def list_documents_tool(conversation_id: str) -> str:
    """List all loaded documents"""
    return get_document_summary(conversation_id)


def clear_documents_tool(conversation_id: str) -> str:
    """Clear all loaded documents"""
    return clear_documents(conversation_id)
