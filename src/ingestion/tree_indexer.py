"""
Deterministic Tree Indexer & Commit DAG Builder (Stage 3)
Provides exact, zero-hallucination directory trees and commit graphs.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import git

from src.config import IGNORE_DIRS, IGNORE_EXTENSIONS


def build_file_tree(repo_path: Path, max_depth: int = 5) -> Dict[str, Any]:
    """
    Builds a recursive hierarchical dictionary representing the repository file tree.
    """
    def _build_subtree(current_path: Path, current_depth: int) -> Dict[str, Any]:
        if current_depth > max_depth:
            return {"type": "directory", "truncated": True}

        tree_node: Dict[str, Any] = {
            "name": current_path.name,
            "type": "directory",
            "path": current_path.relative_to(repo_path).as_posix() if current_path != repo_path else ".",
            "children": []
        }

        try:
            items = sorted(current_path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
            for item in items:
                if item.name.startswith(".") or item.name in IGNORE_DIRS:
                    continue
                if item.is_dir():
                    tree_node["children"].append(_build_subtree(item, current_depth + 1))
                else:
                    if item.suffix.lower() not in IGNORE_EXTENSIONS:
                        tree_node["children"].append({
                            "name": item.name,
                            "type": "file",
                            "path": item.relative_to(repo_path).as_posix(),
                            "extension": item.suffix.lower(),
                            "size_bytes": item.stat().st_size
                        })
        except PermissionError:
            pass

        return tree_node

    return _build_subtree(repo_path, 0)


def format_file_tree_as_text(repo_path: Path, max_depth: int = 4) -> str:
    """
    Renders the repository file hierarchy into an ASCII tree string for LLM or user display.
    """
    lines = [f"📁 {repo_path.name}/"]

    def _render(current_path: Path, prefix: str = "", depth: int = 0):
        if depth >= max_depth:
            lines.append(f"{prefix}└── ... (truncated)")
            return

        try:
            entries = [
                e for e in sorted(current_path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
                if not e.name.startswith(".") and e.name not in IGNORE_DIRS and e.suffix.lower() not in IGNORE_EXTENSIONS
            ]
        except PermissionError:
            return

        for idx, entry in enumerate(entries):
            is_last = (idx == len(entries) - 1)
            connector = "└── " if is_last else "├── "
            child_prefix = "    " if is_last else "│   "

            if entry.is_dir():
                lines.append(f"{prefix}{connector}📁 {entry.name}/")
                _render(entry, prefix + child_prefix, depth + 1)
            else:
                lines.append(f"{prefix}{connector}📄 {entry.name}")

    _render(repo_path, depth=0)
    return "\n".join(lines)


def build_commit_dag(repo_path: Path, max_commits: int = 30) -> List[Dict[str, Any]]:
    """
    Extracts deterministic commit relations (DAG) including parents, SHA, author, and date.
    """
    try:
        repo = git.Repo(repo_path, search_parent_directories=True)
        commits = list(repo.iter_commits(max_count=max_commits))
    except Exception as e:
        print(f"[DAG WARNING] Could not read commits: {e}")
        return []

    dag = []
    for c in commits:
        dag.append({
            "sha": c.hexsha[:8],
            "parents": [p.hexsha[:8] for p in c.parents],
            "author": c.author.name,
            "date": c.committed_datetime.strftime("%Y-%m-%d %H:%M:%S"),
            "message": c.message.strip().split("\n")[0]
        })
    return dag


def format_commit_dag_as_text(repo_path: Path, max_commits: int = 20) -> str:
    """Renders commit history DAG into formatted ASCII tree."""
    dag = build_commit_dag(repo_path, max_commits=max_commits)
    if not dag:
        return "No commit history found."

    lines = ["🌳 Commit History Graph:"]
    for c in dag:
        parents_str = f" (parents: {', '.join(c['parents'])})" if c["parents"] else " (root)"
        lines.append(f"* [{c['sha']}] {c['date']} - {c['author']}: {c['message']}{parents_str}")
    return "\n".join(lines)
