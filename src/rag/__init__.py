"""
RAG Package - Dense embeddings and FAISS vector stores
"""

from src.rag.embeddings import get_embeddings_model
from src.rag.vectorstore import load_or_build_vectorstore

__all__ = ["get_embeddings_model", "load_or_build_vectorstore"]
