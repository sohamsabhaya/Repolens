"""
Codebase Intelligence Agent — CLI Entrypoint
Run multi-turn codebase question answering across Java, Python, TypeScript, Go, Rust, C++, and more.
"""

import argparse
from src.config import PRIMARY_LLM_MODEL
from src.agent.models import get_groq_llm
from src.rag.embeddings import get_embeddings_model
from src.rag.vectorstore import load_or_build_vectorstore
from src.agent.main_graph import create_code_agent


def main():
    parser = argparse.ArgumentParser(
        description="Codebase Intelligence Agent — Multi-Language Grounded Code RAG"
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=".",
        help="GitHub repository URL or local directory path (supports Java, Python, TS, Go, Rust, C++, etc.)",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force rebuild the FAISS vector index cache on disk",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="Explain the main architecture and entrypoint of this project.",
        help="Question to ask about the codebase",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=PRIMARY_LLM_MODEL,
        help="Groq LLM model name (default: qwen/qwen3.8-27b)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=6,
        help="Number of retrieved context chunks (default: 6)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  [*] Codebase Intelligence Agent")
    print("=" * 60)

    # 1. Initialize Models
    print(f"\n[1/4] Initializing Groq LLM ({args.model}) & HuggingFace Embeddings...")
    llm = get_groq_llm(model_name=args.model)
    embeddings = get_embeddings_model()

    # 2. Ingest Repository / Load Cache
    print(f"\n[2/4] Resolving Repository & Vector Store: {args.repo}...")
    vectorstore = load_or_build_vectorstore(
        repo_source=args.repo,
        embeddings=embeddings,
        force_reindex=args.reindex,
    )

    # 3. Compile LangGraph Agent
    print("\n[3/4] Compiling LangGraph Code Reasoning Engine...")
    agent = create_code_agent(llm=llm, vectorstore=vectorstore, top_k=args.top_k)

    # 4. Execute Query
    print(f"\n[4/4] Executing Query: '{args.query}'...\n")
    result = agent.invoke(
        {"question": args.query},
        config={"configurable": {"thread_id": "cli_session_1"}},
    )

    print("=" * 60)
    print("  [ANSWER]")
    print("=" * 60)
    print(result["answer"])

    print("\n" + "-" * 60)
    print("  [CITATIONS]")
    print("-" * 60)
    if result["citations"]:
        for cit in sorted(result["citations"]):
            print(f"  - {cit}")
    else:
        print("  (No direct citations returned)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
