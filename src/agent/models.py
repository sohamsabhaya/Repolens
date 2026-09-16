"""
Groq LLM Client Factory
"""

from typing import Optional
from langchain_groq import ChatGroq
from src.config import GROQ_API_KEY, PRIMARY_LLM_MODEL


def get_groq_llm(
    model_name: Optional[str] = None,
    temperature: float = 0.1,
    streaming: bool = False
) -> ChatGroq:
    """Initializes and returns a ChatGroq LLM client instance."""
    return ChatGroq(
        model=model_name or PRIMARY_LLM_MODEL,
        temperature=temperature,
        api_key=GROQ_API_KEY,
        streaming=streaming
    )
