"""
LangGraph StateGraph Workflow Compiler (Stage 3: Conditional Routing & Deterministic Tree Lookups)
"""

from pathlib import Path
from typing import Dict, Any, Optional
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.agent.state import AgentState
from src.agent.prompts import CODE_SYSTEM_PROMPT
from src.agent.router import classify_query
from src.ingestion.tree_indexer import format_file_tree_as_text, format_commit_dag_as_text


def create_code_agent(
    llm: ChatGroq,
    vectorstore: FAISS,
    repo_path: Optional[Path] = None,
    top_k: int = 6,
    checkpointer: Optional[Any] = None
):
    """
    Compiles and returns the Stage 3 LangGraph agent with conditional routing.
    Graph Structure:
      START -> route_query -> (conditional: 'tree' -> tree_lookup | 'rag' -> retrieve) -> generate -> END
    """

    def route_node(state: AgentState) -> Dict[str, Any]:
        """Classifies the query intent to route execution."""
        question = state["question"]
        intent = classify_query(llm, question)
        return {"route": intent}

    def tree_lookup_node(state: AgentState) -> Dict[str, Any]:
        """Performs deterministic file tree and commit DAG extraction with zero hallucination."""
        target_dir = repo_path or Path.cwd()
        file_tree_text = format_file_tree_as_text(target_dir, max_depth=4)
        commit_dag_text = format_commit_dag_as_text(target_dir, max_commits=15)
        
        combined_structural_context = (
            f"=== DETERMINISTIC FILE TREE ===\n{file_tree_text}\n\n"
            f"=== COMMIT HISTORY DAG ===\n{commit_dag_text}"
        )
        return {
            "tree_context": combined_structural_context,
            "retrieved_docs": []
        }

    def retrieve_node(state: AgentState) -> Dict[str, Any]:
        """Retrieves top-k code and git commit chunks from FAISS."""
        query = state["question"]
        docs = vectorstore.similarity_search(query, k=top_k)
        return {"retrieved_docs": docs}

    def generate_node(state: AgentState) -> Dict[str, Any]:
        """Synthesizes grounded explanation with code, commit, or tree citations."""
        question = state["question"]
        docs = state.get("retrieved_docs", [])
        tree_ctx = state.get("tree_context", "")

        context_parts = []
        citations = []

        if tree_ctx:
            context_parts.append(tree_ctx)
            citations.append("Repository:FileTree & CommitDAG")

        for i, d in enumerate(docs, 1):
            source_type = d.metadata.get("source_type", "code")

            if source_type == "commit":
                sha = d.metadata.get("commit_sha", "unknown")
                author = d.metadata.get("author", "unknown")
                date_val = d.metadata.get("date", "unknown")
                citation_tag = f"commit:{sha} by {author} on {date_val}"
                citations.append(citation_tag)
                context_parts.append(
                    f"--- [Git Commit Document {i}] [{citation_tag}] ---\\n{d.page_content}"
                )
            else:
                fp = d.metadata.get("file_path", "unknown")
                s_line = d.metadata.get("start_line", "?")
                e_line = d.metadata.get("end_line", "?")
                lang = d.metadata.get("language", "code")
                citation_tag = f"{fp}:L{s_line}-L{e_line}"
                citations.append(citation_tag)
                context_parts.append(
                    f"--- [Code Document {i}] [{citation_tag}] (Lang: {lang}) ---\\n{d.page_content}"
                )

        formatted_context = "\n\n".join(context_parts) if context_parts else "No relevant context found."
        messages = [
            {"role": "system", "content": CODE_SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{formatted_context}\n\nQuestion:\n{question}"}
        ]

        response = llm.invoke(messages)
        return {
            "answer": response.content,
            "citations": list(set(citations))
        }

    def route_decision_edge(state: AgentState) -> str:
        """Conditional routing branch."""
        route = state.get("route", "code_rag")
        if route == "tree":
            return "tree_lookup"
        return "retrieve"

    # Assemble StateGraph
    builder = StateGraph(AgentState)
    builder.add_node("route_query", route_node)
    builder.add_node("tree_lookup", tree_lookup_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)

    # Add Edges
    builder.add_edge(START, "route_query")
    builder.add_conditional_edges(
        "route_query",
        route_decision_edge,
        {
            "tree_lookup": "tree_lookup",
            "retrieve": "retrieve"
        }
    )
    builder.add_edge("tree_lookup", "generate")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)

    memory = checkpointer or MemorySaver()
    return builder.compile(checkpointer=memory)
