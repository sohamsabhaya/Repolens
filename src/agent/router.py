"""
Query Router & Intent Classifier (Stage 3)
Classifies queries to direct them to deterministic tree tools, git history, or code RAG.
"""

from typing import Literal
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq


class RouteDecision(BaseModel):
    """Routing classification for codebase query."""
    intent: Literal["tree", "commit_history", "code_rag", "general"] = Field(
        description=(
            "The intended target for the query: "
            "'tree' for directory structure/file listings/folder hierarchy; "
            "'commit_history' for commit logs/diffs/authors/when changes occurred; "
            "'code_rag' for source code logic/functions/classes/how-to; "
            "'general' for high-level repository overviews or mixed queries."
        )
    )
    reasoning: str = Field(default="", description="Brief reason for the routing choice.")


# Alias for backward compatibility
RouteQuery = RouteDecision


def classify_query(llm: ChatGroq, question: str) -> str:
    """
    Classifies a user query into one of: 'tree', 'commit_history', 'code_rag', 'general'.
    """
    prompt = (
        "Classify the following user query about a software repository into exactly one category:\n"
        "- 'tree': If asking for directory structure, file lists, folder contents, or where files live.\n"
        "- 'commit_history': If asking about commits, who changed what, git history, commit messages, or diffs.\n"
        "- 'code_rag': If asking about implementation details, functions, classes, code behavior, or how code works.\n"
        "- 'general': For general questions, broad architecture summaries, or open inquiries.\n\n"
        f"Query: \"{question}\"\n\n"
        "Respond with ONLY one word: tree, commit_history, code_rag, or general."
    )

    try:
        response = llm.invoke([{"role": "user", "content": prompt}])
        raw_intent = response.content.strip().lower()
        for valid in ["tree", "commit_history", "code_rag", "general"]:
            if valid in raw_intent:
                return valid
        return "code_rag"
    except Exception:
        return "code_rag"
