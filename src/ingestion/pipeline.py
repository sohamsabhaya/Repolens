"""
Ingestion Pipeline - Repository Cloning, Git History Extraction & Parallel File Processing
"""

import os
import re
import hashlib
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import git
from langchain_core.documents import Document

from src.config import (
    CLONED_REPOS_DIR,
    INDEXES_DIR,
    IGNORE_DIRS,
    IGNORE_EXTENSIONS,
    DEFAULT_MAX_WORKERS,
)
from src.ingestion.code_parser import UniversalCodeChunker
from src.ingestion.git_parser import GitHistoryExtractor


def get_repo_cache_dir(repo_source: str, base_cache_dir: Optional[Path] = None) -> Path:
    """Computes a deterministic cache directory for a repository."""
    cache_base = base_cache_dir or INDEXES_DIR
    repo_hash = hashlib.sha256(repo_source.strip().encode("utf-8")).hexdigest()[:12]
    clean_name = re.sub(r'[^a-zA-Z0-9_-]', '_', repo_source.split('/')[-1].replace('.git', ''))
    return cache_base / f"{clean_name}_{repo_hash}"


def clone_or_load_repo(repo_source: str, target_dir: Optional[Path] = None) -> Path:
    """Clones a remote repository URL or resolves a local repository path."""
    if repo_source.startswith(("http://", "https://", "git@")):
        repo_name = repo_source.rstrip("/").split("/")[-1].replace(".git", "")
        dest_path = target_dir or (CLONED_REPOS_DIR / repo_name)
        
        if dest_path.exists():
            print(f"[LOCAL REPO DETECTED] Using cached clone at {dest_path.resolve()}")
            return dest_path
            
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[CLONING] Cloning {repo_source} (depth=50)...")
        git.Repo.clone_from(repo_source, str(dest_path), depth=50)
        print(f"[CLONING COMPLETE] {dest_path.resolve()}")
        return dest_path
    else:
        local_path = Path(repo_source)
        if not local_path.exists():
            raise FileNotFoundError(f"Local repository path does not exist: {local_path.resolve()}")
        print(f"[LOCAL PATH LOADED] {local_path.resolve()}")
        return local_path


def parallel_scan_and_chunk(
    repo_path: Path,
    include_git_history: bool = True,
    max_commits: int = 50,
    max_workers: int = DEFAULT_MAX_WORKERS
) -> List[Document]:
    """
    Walks the repository, chunks source files concurrently, and extracts Git history / diffs.
    """
    candidate_files: List[Path] = []
    
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for f in files:
            file_p = Path(root) / f
            ext = file_p.suffix.lower()
            if ext not in IGNORE_EXTENSIONS and not f.startswith("."):
                candidate_files.append(file_p)

    print(f"[PARSING] Scanning {len(candidate_files)} files using {max_workers} threads...")

    all_documents: List[Document] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(UniversalCodeChunker.process_file, fp, repo_path) for fp in candidate_files]
        for future in as_completed(futures):
            res = future.result()
            if res:
                all_documents.extend(res)

    print(f"[PARSING COMPLETE] Generated {len(all_documents)} semantic code chunks.")

    # Ingest Git Commit History & Diffs (Stage 2)
    if include_git_history:
        print(f"[GIT INGESTION] Extracting up to {max_commits} commits with diffs...")
        commit_docs = GitHistoryExtractor.extract_commits(repo_path, max_commits=max_commits)
        all_documents.extend(commit_docs)
        print(f"[GIT COMPLETE] Added {len(commit_docs)} commit history records.")

    if not all_documents:
        raise ValueError(f"No parseable code or configuration files found in {repo_path}!")

    return all_documents
