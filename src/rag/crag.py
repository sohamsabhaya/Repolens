"""
CampusX Corrective RAG (CRAG) Engine (Stage 4)
Confidence threshold grading (0.7 / 0.3), knowledge strips decomposition, query rewriting, and fallback.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.documents import Document

UPPER_CONFIDENCE_THRESHOLD: float = 0.7
LOWER_CONFIDENCE_THRESHOLD: float = 0.3


class DocEvalScore(BaseModel):
    """Document relevance evaluation score from CRAG evaluator."""
    score: float = Field(
        description="Relevance confidence score between 0.0 and 1.0 (>= 0.7 Correct, 0.3-0.7 Ambiguous, < 0.3 Incorrect)"
    )
    reasoning: str = Field(
        default="",
        description="Explanation for the relevance grading."
    )


def decompose_into_knowledge_strips(doc_text: str, strip_size: int = 300) -> List[str]:
    """
    Decomposes a retrieved document chunk into fine-grained knowledge strips (sentences/paragraphs)
    to filter out noise before generation.
    """
    lines = [line.strip() for line in doc_text.split("\n") if line.strip()]
    strips = []
    current_strip = []
    current_len = 0
    
    for line in lines:
        current_strip.append(line)
        current_len += len(line)
        if current_len >= strip_size:
            strips.append("\n".join(current_strip))
            current_strip = []
            current_len = 0
            
    if current_strip:
        strips.append("\n".join(current_strip))
        
    return strips or [doc_text]


def evaluate_doc_relevance(
    query: str,
    doc: Document,
    evaluator_llm: Any,
) -> DocEvalScore:
    """
    Evaluates document relevance using strict CampusX CRAG threshold rubric.
    """
    structured_llm = evaluator_llm.with_structured_output(DocEvalScore)
    prompt = f"""You are a strict Retrieval Evaluator for Codebase RAG.
Query: "{query}"

Document Excerpt:
{doc.page_content[:1500]}

Score relevance from 0.0 to 1.0:
- >= 0.7: Contains direct, correct answers/code logic for the query.
- 0.3 to 0.7: Partial or ambiguous match requiring query refinement.
- < 0.3: Irrelevant or completely unrelated.
"""
    try:
        return structured_llm.invoke(prompt)
    except Exception:
        # Default neutral score
        return DocEvalScore(score=0.5, reasoning="Evaluator fallback")
