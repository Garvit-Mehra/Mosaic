"""
Mosaic LLM Provider

Supports multiple backends via a single configuration:
- ollama:    Local Ollama server (default for dev)
- openai:    OpenAI API (GPT-4, etc.)
- compatible: Any OpenAI-compatible API (vLLM, TGI, Groq, Together, LiteLLM)

Controlled by environment variables:
  LLM_PROVIDER=ollama|openai|compatible
  LLM_MODEL=mistral (or gpt-4o, llama-3.1-8b, etc.)
  LLM_BASE_URL=http://localhost:11434 (for ollama/compatible)
  LLM_API_KEY=sk-... (for openai/compatible)
"""

import os
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from utils.logger import get_logger

logger = get_logger("llm")


def get_chat_model(
    model: Optional[str] = None,
    temperature: float = 0.0,
    reasoning: bool = False,
):
    """
    Get a LangChain chat model based on environment configuration.
    
    Priority:
    1. Explicit `model` parameter
    2. LLM_MODEL env var
    3. Default: "llama3.2"
    
    Returns a LangChain BaseChatModel instance.
    """
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    model_name = model or os.getenv("LLM_MODEL", "llama3.2")
    base_url = os.getenv("LLM_BASE_URL")
    api_key = os.getenv("LLM_API_KEY", "")

    if provider == "ollama":
        from langchain_ollama.chat_models import ChatOllama
        kwargs = {"model": model_name, "temperature": temperature}
        if base_url:
            kwargs["base_url"] = base_url
        # Only pass reasoning if the model supports it
        try:
            return ChatOllama(**kwargs, reasoning=reasoning)
        except TypeError:
            return ChatOllama(**kwargs)

    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        kwargs = {
            "model": model_name,
            "temperature": temperature,
            "api_key": api_key or os.getenv("OPENAI_API_KEY"),
        }
        if base_url:
            kwargs["base_url"] = base_url
        return ChatOpenAI(**kwargs)

    elif provider == "compatible":
        # Any OpenAI-compatible API: vLLM, TGI, Groq, Together, LiteLLM, etc.
        from langchain_openai import ChatOpenAI
        if not base_url:
            raise ValueError(
                "LLM_PROVIDER=compatible requires LLM_BASE_URL to be set. "
                "Example: LLM_BASE_URL=http://localhost:8000/v1"
            )
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key or "not-needed",  # Some local servers don't need a key
            base_url=base_url,
        )

    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER: {provider}. "
            f"Valid options: ollama, openai, compatible"
        )


def get_classifier_model():
    """Get a fast model for agent classification (low temperature, no reasoning)."""
    return get_chat_model(temperature=0.0, reasoning=False)


def get_agent_model():
    """Get the model used by agents for generating responses."""
    return get_chat_model(
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
        reasoning=False,
    )
