"""
State Definitions & Schemas for LangGraph Agent (Stage 3)
"""

from typing import List, Optional, TypedDict
from pydantic import BaseModel, Field
from langchain_core.documents import Document


class Citation(BaseModel):
    file_path: str = Field(description="Path of the cited file.")
    lines: Optional[str] = Field(default=None, description="Line numbers cited, e.g. L10-L45.")


class AgentState(TypedDict):
    question: str
    route: Optional[str]
    retrieved_docs: List[Document]
    tree_context: Optional[str]
    answer: str
    citations: List[str]
