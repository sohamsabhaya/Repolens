"""
Git History Parser - Commit Metadata & Diff Hunk Extraction
"""

from pathlib import Path
from typing import List, Optional
import git
from langchain_core.documents import Document


class GitHistoryExtractor:
    """Extracts git commit history, messages, changed files, and diff patches for semantic search."""

    IGNORE_DIFF_EXTENSIONS = {
        ".lock", ".json", ".svg", ".png", ".jpg", ".jpeg", ".ico",
        ".min.js", ".min.css", ".map", ".sum", ".mod"
    }
    MAX_DIFF_CHARS = 2000

    @classmethod
    def extract_commits(cls, repo_path: Path, max_commits: int = 50) -> List[Document]:
        """
        Parses git commit log into structured LangChain Documents.
        """
        try:
            repo = git.Repo(repo_path, search_parent_directories=True)
            commits = list(repo.iter_commits(max_count=max_commits))
        except Exception as e:
            print(f"[GIT WARNING] Could not read git history from {repo_path}: {e}")
            return []

        commit_docs: List[Document] = []
        for commit in commits:
            sha_short = commit.hexsha[:8]
            author = f"{commit.author.name} <{commit.author.email}>"
            date_str = commit.committed_datetime.strftime("%Y-%m-%d %H:%M:%S")
            message = commit.message.strip()

            # Changed files
            try:
                stats = commit.stats.files
                changed_files = list(stats.keys())
            except Exception:
                changed_files = []

            # Diff patch
            diff_text = ""
            if commit.parents:
                try:
                    parent = commit.parents[0]
                    diffs = parent.diff(commit, create_patch=True)
                    diff_chunks = []
                    for d in diffs:
                        if any(d.a_path and d.a_path.endswith(ext) for ext in cls.IGNORE_DIFF_EXTENSIONS):
                            continue
                        if d.diff:
                            patch_content = d.diff.decode("utf-8", errors="ignore")[:500]
                            diff_chunks.append(f"--- {d.a_path} ---\n{patch_content}")
                    diff_text = "\n".join(diff_chunks)[:cls.MAX_DIFF_CHARS]
                except Exception:
                    diff_text = ""

            doc_content = (
                f"Commit SHA: {sha_short}\n"
                f"Author: {author}\n"
                f"Date: {date_str}\n"
                f"Commit Message: {message}\n"
                f"Files Modified: {', '.join(changed_files) if changed_files else 'None'}\n\n"
                f"Diff Summary / Patch Snippets:\n{diff_text if diff_text else 'No patch available.'}"
            )

            commit_docs.append(
                Document(
                    page_content=doc_content,
                    metadata={
                        "source_type": "commit",
                        "commit_sha": sha_short,
                        "author": commit.author.name,
                        "date": date_str,
                        "message": message[:80],
                    }
                )
            )

        print(f"[GIT PARSER] Extracted {len(commit_docs)} commit history documents.")
        return commit_docs

    @classmethod
    def get_detailed_commit_records(cls, repo_path: Path, max_commits: int = 50) -> List[Dict]:
        """
        Extracts rich commit objects with stats, files changed, and per-file diffs for visual UI cards.
        """
        try:
            repo = git.Repo(repo_path, search_parent_directories=True)
            commits = list(repo.iter_commits(max_count=max_commits))
        except Exception as e:
            print(f"[GIT WARNING] Could not read detailed commits: {e}")
            return []

        detailed_records = []
        for commit in commits:
            sha_short = commit.hexsha[:8]
            author_name = commit.author.name
            author_email = commit.author.email
            date_str = commit.committed_datetime.strftime("%b %d, %Y %H:%M")
            message = commit.message.strip()

            stats_dict = {}
            try:
                stats = commit.stats
                total_insertions = stats.total.get("insertions", 0)
                total_deletions = stats.total.get("deletions", 0)
                stats_dict = stats.files
            except Exception:
                total_insertions = 0
                total_deletions = 0

            # Generate structured diff per file
            file_diffs = []
            if commit.parents:
                try:
                    parent = commit.parents[0]
                    diffs = parent.diff(commit, create_patch=True)
                    for d in diffs:
                        if d.diff:
                            patch = d.diff.decode("utf-8", errors="ignore")
                            file_diffs.append({
                                "file": d.b_path or d.a_path,
                                "change_type": d.change_type,
                                "patch": patch,
                            })
                except Exception:
                    pass

            detailed_records.append({
                "sha": sha_short,
                "full_sha": commit.hexsha,
                "author": author_name,
                "email": author_email,
                "date": date_str,
                "message": message,
                "total_insertions": total_insertions,
                "total_deletions": total_deletions,
                "files_stats": stats_dict,
                "diffs": file_diffs,
            })

        return detailed_records
