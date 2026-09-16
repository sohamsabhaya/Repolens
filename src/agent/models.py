"""
Centralized Model & Embeddings Factory (File 1)
Manages Groq LLMs (Reasoning & Fast Router) and Local HuggingFace Embeddings.
"""

from typing import Optional
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

import src.config as config


def get_reasoning_llm(
    model_name: Optional[str] = None,
    temperature: float = 0.1,
    streaming: bool = False
) -> ChatGroq:
    """Primary LLM for code analysis, explanation, and grounded generation."""
    groq_key = getattr(config, "GROQ_API_KEY", None)
    model = model_name or getattr(config, "PRIMARY_LLM_MODEL", "qwen/qwen3.8-27b")
    return ChatGroq(
        model=model,
        temperature=temperature,
        api_key=groq_key,
        streaming=streaming,
    )


def get_fast_router_llm() -> ChatGroq:
    """Fast, low-latency LLM for intent classification and conditional routing."""
    groq_key = getattr(config, "GROQ_API_KEY", None)
    model = getattr(config, "FAST_LLM_MODEL", "qwen/qwen3.8-27b")
    return ChatGroq(
        model=model,
        temperature=0.0,
        api_key=groq_key,
    )


def get_embeddings_model(
    model_name: Optional[str] = None,
    device: str = "cpu"
) -> HuggingFaceEmbeddings:
    """Local dense embedding model (zero cost, 100% offline)."""
    emb_model = model_name or getattr(config, "DEFAULT_EMBEDDING_MODEL", None) or getattr(config, "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    return HuggingFaceEmbeddings(
        model_name=emb_model,
        model_kwargs={"device": device},
    )


# Backward compatibility alias
get_groq_llm = get_reasoning_llm
