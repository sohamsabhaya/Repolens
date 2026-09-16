"""
Multi-Language Code & Text Chunker
Supports Python AST parsing, syntax-aware splitting for Java, JS/TS, Go, Rust, C++, C#, Kotlin, Scala,
and structured configuration/documentation chunking.
"""

import ast
from pathlib import Path
from typing import List, Dict
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    VALID_CONFIG_EXTENSIONS,
)

# Extension to LangChain Language Grammar Mapping
EXTENSION_TO_LANGUAGE: Dict[str, Language] = {
    # JVM Languages
    ".java": Language.JAVA,
    ".kt": Language.KOTLIN,
    ".kts": Language.KOTLIN,
    ".scala": Language.SCALA,

    # Python
    ".py": Language.PYTHON,

    # JavaScript / TypeScript Ecosystem
    ".js": Language.JS,
    ".jsx": Language.JS,
    ".ts": Language.TS,
    ".tsx": Language.TS,
    ".mjs": Language.JS,
    ".cjs": Language.JS,

    # Systems & Compiled Languages
    ".go": Language.GO,
    ".rs": Language.RUST,
    ".cpp": Language.CPP,
    ".cxx": Language.CPP,
    ".cc": Language.CPP,
    ".c": Language.C,
    ".h": Language.C,
    ".hpp": Language.CPP,
    ".cs": Language.CSHARP,

    # Web & Scripting
    ".php": Language.PHP,
    ".rb": Language.RUBY,
    ".html": Language.HTML,
    ".htm": Language.HTML,
    ".md": Language.MARKDOWN,
}

# Pre-instantiated Language Splitters
LANGUAGE_SPLITTERS = {
    lang: RecursiveCharacterTextSplitter.from_language(
        language=lang,
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    )
    for lang in set(EXTENSION_TO_LANGUAGE.values())
}

DEFAULT_TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", " ", ""],
)


class UniversalCodeChunker:
    """
    Intelligently splits source code files into semantic chunks with file paths,
    language tags, and line number ranges.
    """

    @staticmethod
    def chunk_python(content: str, relative_path: str) -> List[Document]:
        """Deep AST chunking for Python files."""
        lines = content.splitlines()
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return UniversalCodeChunker.chunk_generic_code(content, relative_path, Language.PYTHON)

        chunks: List[Document] = []
        module_doc = ast.get_docstring(tree)
        if module_doc:
            chunks.append(Document(
                page_content=f"File: {relative_path} (Module Docstring)\n\n{module_doc}",
                metadata={
                    "file_path": relative_path,
                    "chunk_type": "module_doc",
                    "name": relative_path,
                    "start_line": 1,
                    "end_line": len(module_doc.splitlines()),
                    "language": "python",
                    "source_type": "code"
                }
            ))

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start_l = node.lineno
                end_l = getattr(node, "end_lineno", start_l + len(ast.unparse(node).splitlines()))
                code_segment = "\n".join(lines[start_l - 1:end_l])
                docstring = ast.get_docstring(node) or "No docstring provided."
                
                formatted_content = f"File: {relative_path} (Lines {start_l}-{end_l})\nFunction: {node.name}\nDocstring: {docstring}\n\nCode:\n{code_segment}"
                chunks.append(Document(
                    page_content=formatted_content,
                    metadata={
                        "file_path": relative_path,
                        "chunk_type": "function",
                        "name": node.name,
                        "start_line": start_l,
                        "end_line": end_l,
                        "language": "python",
                        "source_type": "code"
                    }
                ))

            elif isinstance(node, ast.ClassDef):
                start_l = node.lineno
                end_l = getattr(node, "end_lineno", start_l + len(ast.unparse(node).splitlines()))
                code_segment = "\n".join(lines[start_l - 1:end_l])
                docstring = ast.get_docstring(node) or "No docstring provided."
                methods = [m.name for m in node.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]

                formatted_content = f"File: {relative_path} (Lines {start_l}-{end_l})\nClass: {node.name}\nMethods: {', '.join(methods) if methods else 'None'}\nDocstring: {docstring}\n\nCode:\n{code_segment}"
                chunks.append(Document(
                    page_content=formatted_content,
                    metadata={
                        "file_path": relative_path,
                        "chunk_type": "class",
                        "name": node.name,
                        "start_line": start_l,
                        "end_line": end_l,
                        "language": "python",
                        "source_type": "code"
                    }
                ))

        if not chunks:
            return UniversalCodeChunker.chunk_generic_code(content, relative_path, Language.PYTHON)

        return chunks

    @staticmethod
    def chunk_generic_code(content: str, relative_path: str, language: Language) -> List[Document]:
        """Syntax-aware chunking for Java, JS/TS, Go, Rust, C++, C#, Kotlin, etc."""
        splitter = LANGUAGE_SPLITTERS.get(language, DEFAULT_TEXT_SPLITTER)
        text_chunks = splitter.split_text(content)
        lines = content.splitlines()
        docs: List[Document] = []

        current_line = 1
        for idx, chunk in enumerate(text_chunks):
            chunk_line_count = len(chunk.splitlines())
            start_l = current_line
            end_l = min(len(lines), start_l + chunk_line_count - 1)
            current_line = max(1, end_l - 2)

            docs.append(Document(
                page_content=f"File: {relative_path} (Lines {start_l}-{end_l})\nLanguage: {language.value}\n\nCode:\n{chunk}",
                metadata={
                    "file_path": relative_path,
                    "chunk_type": "code_block",
                    "name": f"{relative_path}#part{idx+1}",
                    "start_line": start_l,
                    "end_line": end_l,
                    "language": language.value,
                    "source_type": "code"
                }
            ))
        return docs

    @staticmethod
    def chunk_config(content: str, relative_path: str) -> List[Document]:
        """Chunk configuration and documentation files."""
        text_chunks = DEFAULT_TEXT_SPLITTER.split_text(content)
        lines = content.splitlines()
        docs: List[Document] = []
        for idx, chunk in enumerate(text_chunks):
            docs.append(Document(
                page_content=f"File: {relative_path}\n\n{chunk}",
                metadata={
                    "file_path": relative_path,
                    "chunk_type": "config_doc",
                    "name": relative_path,
                    "start_line": 1,
                    "end_line": len(lines),
                    "language": "config",
                    "source_type": "config"
                }
            ))
        return docs

    @classmethod
    def process_file(cls, file_path: Path, repo_root: Path) -> List[Document]:
        """Processes a single file and dispatches to the corresponding language chunker."""
        try:
            rel_path = file_path.relative_to(repo_root).as_posix()
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            if not content.strip():
                return []

            ext = file_path.suffix.lower()

            if ext == ".py":
                return cls.chunk_python(content, rel_path)
            elif ext in EXTENSION_TO_LANGUAGE:
                return cls.chunk_generic_code(content, rel_path, EXTENSION_TO_LANGUAGE[ext])
            elif ext in VALID_CONFIG_EXTENSIONS or file_path.name in {
                "Dockerfile", "Makefile", "pom.xml", "build.gradle", "settings.gradle"
            }:
                return cls.chunk_config(content, rel_path)
            return []
        except Exception as e:
            print(f"Warning: Failed to parse {file_path}: {e}")
            return []
