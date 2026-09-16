"""
Human-in-the-Loop (HITL) & Query Disambiguation Module for RepoLens.
Detects ambiguous references in queries (multiple matching files/functions)
and leverages LangGraph interrupt() and Command(resume=...) to pause execution and prompt the user.
"""

from typing import List, Dict, Any, Optional
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import interrupt, Command
from pydantic import BaseModel, Field

from src.agent.models import get_fast_router_llm


class AmbiguityAnalysis(BaseModel):
    """Structured evaluation of whether a user query contains ambiguity across codebase entities."""
    is_ambiguous: bool = Field(
        description="True if the query refers to a name/function/class/file that exists in multiple places."
    )
    candidates: List[str] = Field(
        default_factory=list,
        description="List of candidate files, functions, or interpretations that the user might mean."
    )
    clarification_question: str = Field(
        default="",
        description="A polite question asking the user which candidate they want to analyze."
    )


def detect_codebase_ambiguity(
    query: str,
    retrieved_docs: List[Any],
    llm: Optional[Any] = None,
) -> AmbiguityAnalysis:
    """
    Analyzes retrieved documents to determine if the user query refers to multiple distinct entities.
    """
    if not retrieved_docs or len(retrieved_docs) < 2:
        return AmbiguityAnalysis(is_ambiguous=False, candidates=[], clarification_question="")

    if llm is None:
        llm = get_fast_router_llm()

    # Extract unique source files or symbol names from retrieved docs
    unique_sources = set()
    for doc in retrieved_docs:
        src = doc.metadata.get("source_file") or doc.metadata.get("file_path") or doc.metadata.get("source")
        if src:
            unique_sources.add(src)

    if len(unique_sources) <= 1:
        return AmbiguityAnalysis(is_ambiguous=False, candidates=[], clarification_question="")

    # LLM structured prompt to assess genuine ambiguity vs multi-file context
    structured_llm = llm.with_structured_output(AmbiguityAnalysis)
    sources_text = "\n".join([f"- {s}" for s in list(unique_sources)[:6]])

    prompt = f"""You are a Codebase Ambiguity Detector for an AI Code Assistant.
User Query: "{query}"

Retrieved relevant files/components:
{sources_text}

Determine if the user's query is ambiguous because the entity they asked about exists in multiple distinct files (e.g. multiple implementations of `authenticate`, `config`, `User`, etc.), OR if they are asking a broad question that naturally spans multiple files.

If genuinely ambiguous (user likely meant ONE specific file/function out of several candidates):
- Set is_ambiguous = True
- List candidates = [list of file paths or interpretations]
- Provide clarification_question = "I found multiple matching components. Which one did you mean?"

If NOT ambiguous:
- Set is_ambiguous = False
"""
    try:
        return structured_llm.invoke(prompt)
    except Exception as e:
        print(f"[HITL AMBIGUITY EVAL ERROR] {e}")
        return AmbiguityAnalysis(is_ambiguous=False, candidates=[], clarification_question="")


def hitl_disambiguation_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node that pauses execution with interrupt() when ambiguity is detected.
    Resumes with the user's selected candidate choice.
    """
    ambiguity: Optional[AmbiguityAnalysis] = state.get("ambiguity_analysis")
    
    if ambiguity and ambiguity.is_ambiguous and ambiguity.candidates:
        # Halt execution and prompt the user
        interrupt_payload = {
            "type": "disambiguation_required",
            "question": ambiguity.clarification_question,
            "candidates": ambiguity.candidates,
        }
        
        # interrupt() pauses the graph and saves state to checkpointer
        user_choice = interrupt(interrupt_payload)
        
        # When resumed via Command(resume="..."), user_choice holds the selected option
        return {
            "selected_entity": user_choice,
            "messages": [AIMessage(content=f"Proceeding with analysis for selected component: `{user_choice}`")]
        }

    return {}
