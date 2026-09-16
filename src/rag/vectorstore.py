"""
Vector Store Management & Disk Caching with FAISS
"""

from pathlib import Path
from typing import Optional
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import DEFAULT_MAX_WORKERS
from src.ingestion.pipeline import (
    get_repo_cache_dir,
    clone_or_load_repo,
    parallel_scan_and_chunk,
)


def load_or_build_vectorstore(
    repo_source: str,
    embeddings: HuggingFaceEmbeddings,
    target_clone_dir: Optional[Path] = None,
    force_reindex: bool = False,
    max_workers: int = DEFAULT_MAX_WORKERS
) -> FAISS:
    """
    Checks if a vector index exists on disk.
    If cached and not force_reindex, loads instantly (<0.2s).
    Otherwise, clones/loads the repo, chunks files concurrently, and builds the FAISS store.
    """
    cache_dir = get_repo_cache_dir(repo_source)

    # 1. Fast Cache Hit Check
    if not force_reindex and cache_dir.exists() and (cache_dir / "index.faiss").exists():
        print(f"[CACHE HIT] Loading existing FAISS index from {cache_dir.resolve()} (zero lag)...")
        return FAISS.load_local(str(cache_dir), embeddings, allow_dangerous_deserialization=True)

    # 2. Ingest Repository
    repo_path = clone_or_load_repo(repo_source, target_dir=target_clone_dir)
    documents = parallel_scan_and_chunk(repo_path, max_workers=max_workers)

    # 3. Build & Persist FAISS Index
    print("[EMBEDDING] Encoding chunks & constructing FAISS vector index...")
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(cache_dir))
    print(f"[INDEX PERSISTED] Saved FAISS vector store to {cache_dir.resolve()}")

    return vectorstore
