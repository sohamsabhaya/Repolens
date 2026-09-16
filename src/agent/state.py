"""
State Definitions & Schemas for LangGraph Agent
"""

from typing import List, Optional, TypedDict
from pydantic import BaseModel, Field
from langchain_core.documents import Document


class Citation(BaseModel):
    file_path: str = Field(description="Path of the cited file.")
    lines: Optional[str] = Field(default=None, description="Line numbers cited, e.g. L10-L45.")


class AgentState(TypedDict):
    question: str
    retrieved_docs: List[Document]
    answer: str
    citations: List[str]
