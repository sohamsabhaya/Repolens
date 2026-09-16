"""
Ingestion Package - Multi-language parsers and repository ingestion pipelines
"""

from src.ingestion.code_parser import UniversalCodeChunker
from src.ingestion.pipeline import clone_or_load_repo, parallel_scan_and_chunk

__all__ = ["UniversalCodeChunker", "clone_or_load_repo", "parallel_scan_and_chunk"]
