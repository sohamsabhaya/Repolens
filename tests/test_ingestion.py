"""
Unit tests for multi-language chunking and ingestion
"""

import sys
import tempfile
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.code_parser import UniversalCodeChunker
from src.ingestion.pipeline import parallel_scan_and_chunk


def test_python_ast_chunking():
    py_code = '''"""Sample module docstring."""

class PaymentService:
    """Handles payments."""
    def process(self, amount: float) -> bool:
        return amount > 0

def calculate_tax(subtotal: float) -> float:
    """Calculates tax."""
    return subtotal * 0.18
'''
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        sample_file = tmp_path / "payment.py"
        sample_file.write_text(py_code, encoding="utf-8")

        docs = UniversalCodeChunker.process_file(sample_file, tmp_path)
        assert len(docs) >= 2
        chunk_types = [d.metadata["chunk_type"] for d in docs]
        assert "function" in chunk_types or "class" in chunk_types
        assert any(d.metadata.get("language") == "python" for d in docs)
        print("[PASS] Python AST chunking test passed.")


def test_java_code_chunking():
    java_code = '''package com.example.service;

import org.springframework.stereotype.Service;

@Service
public class UserService {
    public String getUserName(Long id) {
        return "User_" + id;
    }
}
'''
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        sample_file = tmp_path / "UserService.java"
        sample_file.write_text(java_code, encoding="utf-8")

        docs = UniversalCodeChunker.process_file(sample_file, tmp_path)
        assert len(docs) >= 1
        assert docs[0].metadata["language"] == "java"
        assert "UserService" in docs[0].page_content
        print("[PASS] Java code chunking test passed.")


def test_parallel_scan():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("def hello(): pass\n", encoding="utf-8")
        (tmp_path / "src" / "App.java").write_text("public class App {}\n", encoding="utf-8")
        (tmp_path / "README.md").write_text("# Project Docs\nHello world", encoding="utf-8")

        docs = parallel_scan_and_chunk(tmp_path, max_workers=4)
        assert len(docs) >= 3
        print("[PASS] Parallel repository scan test passed.")


if __name__ == "__main__":
    print("\n--- Running Ingestion Tests ---")
    test_python_ast_chunking()
    test_java_code_chunking()
    test_parallel_scan()
    print("\nAll Ingestion Tests Passed Successfully!")
