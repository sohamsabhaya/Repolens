"""
Embedding Model Factory
"""

from typing import Optional
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import EMBEDDING_MODEL_NAME


def get_embeddings_model(
    model_name: Optional[str] = None,
    device: str = "cpu"
) -> HuggingFaceEmbeddings:
    """Initializes and returns a local HuggingFace embedding model."""
    name = model_name or EMBEDDING_MODEL_NAME
    return HuggingFaceEmbeddings(
        model_name=name,
        model_kwargs={"device": device}
    )
