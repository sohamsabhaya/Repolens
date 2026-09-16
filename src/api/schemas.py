"""
Pydantic Schemas for RepoLens API
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    repo_source: str = Field(..., description="GitHub URL or local repository path")
    force_reindex: bool = Field(default=False, description="Whether to rebuild FAISS vectorstore")


class IngestResponse(BaseModel):
    status: str
    repo_source: str
    resolved_path: str
    total_docs_indexed: int
    message: str


class ChatRequest(BaseModel):
    query: str = Field(..., description="User query or instruction")
    thread_id: Optional[str] = Field(default=None, description="UUID session identifier")


class ChatResponse(BaseModel):
    thread_id: str
    response: str
    intent: Optional[str] = None
    citations: List[Dict[str, Any]] = []
    is_interrupted: bool = False
    interrupt_payload: Optional[Dict[str, Any]] = None


class ResumeRequest(BaseModel):
    thread_id: str = Field(..., description="Thread ID of the interrupted session")
    resume_value: str = Field(..., description="Selected candidate or user answer to resume with")


class TreeResponse(BaseModel):
    repo_path: str
    tree_json: Dict[str, Any]
    formatted_tree: str


class CommitDagResponse(BaseModel):
    repo_path: str
    commit_count: int
    commits: List[Dict[str, Any]]
    formatted_dag: str
