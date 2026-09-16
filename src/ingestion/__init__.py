"""
Ingestion Package - Multi-language parsers, Git history extractors, File Tree Indexers & Pipelines
"""

from src.ingestion.code_parser import UniversalCodeChunker
from src.ingestion.git_parser import GitHistoryExtractor
from src.ingestion.tree_indexer import (
    build_file_tree,
    format_file_tree_as_text,
    build_commit_dag,
    format_commit_dag_as_text,
)
from src.ingestion.pipeline import clone_or_load_repo, parallel_scan_and_chunk

__all__ = [
    "UniversalCodeChunker",
    "GitHistoryExtractor",
    "build_file_tree",
    "format_file_tree_as_text",
    "build_commit_dag",
    "format_commit_dag_as_text",
    "clone_or_load_repo",
    "parallel_scan_and_chunk",
]
