"""
LangGraph StateGraph Workflow Compiler
"""

from typing import Dict, Any, Optional
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.agent.state import AgentState
from src.agent.prompts import CODE_SYSTEM_PROMPT


def create_code_agent(
    llm: ChatGroq,
    vectorstore: FAISS,
    top_k: int = 6,
    checkpointer: Optional[Any] = None
):
    """
    Compiles and returns the LangGraph code RAG agent.
    Linear Workflow: START -> retrieve -> generate -> END
    """

    def retrieve_node(state: AgentState) -> Dict[str, Any]:
        """Retrieves top-k code and doc chunks from FAISS."""
        query = state["question"]
        docs = vectorstore.similarity_search(query, k=top_k)
        return {"retrieved_docs": docs}

    def generate_node(state: AgentState) -> Dict[str, Any]:
        """Synthesizes grounded explanation with inline citations."""
        question = state["question"]
        docs = state.get("retrieved_docs", [])

        context_parts = []
        citations = []
        for i, d in enumerate(docs, 1):
            fp = d.metadata.get("file_path", "unknown")
            s_line = d.metadata.get("start_line", "?")
            e_line = d.metadata.get("end_line", "?")
            lang = d.metadata.get("language", "text")
            citation_tag = f"{fp}:L{s_line}-L{e_line}"
            citations.append(citation_tag)
            context_parts.append(
                f"--- Document {i} [{citation_tag}] (Lang: {lang}) ---\n{d.page_content}"
            )

        formatted_context = "\n\n".join(context_parts)
        messages = [
            {"role": "system", "content": CODE_SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{formatted_context}\n\nQuestion:\n{question}"}
        ]

        response = llm.invoke(messages)
        return {
            "answer": response.content,
            "citations": list(set(citations))
        }

    builder = StateGraph(AgentState)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)

    memory = checkpointer or MemorySaver()
    return builder.compile(checkpointer=memory)
