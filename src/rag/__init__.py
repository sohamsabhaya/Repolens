"""
RAG Package - Dense embeddings and FAISS vector stores
"""

from src.rag.embeddings import get_embeddings_model
from src.rag.vectorstore import load_or_build_vectorstore
from src.rag.crag import (
    DocEvalScore,
    UPPER_CONFIDENCE_THRESHOLD,
    LOWER_CONFIDENCE_THRESHOLD,
    decompose_into_knowledge_strips,
    evaluate_doc_relevance,
)

__all__ = [
    "get_embeddings_model",
    "load_or_build_vectorstore",
    "DocEvalScore",
    "UPPER_CONFIDENCE_THRESHOLD",
    "LOWER_CONFIDENCE_THRESHOLD",
    "decompose_into_knowledge_strips",
    "evaluate_doc_relevance",
]
