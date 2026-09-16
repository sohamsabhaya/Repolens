"""
RepoLens — Explorer & Streaming Studio Routes
Provides stats, file inspection, and streaming LLM reasoning endpoints for the Web UI.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.agent.models import get_reasoning_llm
from src.ingestion.git_parser import GitHistoryExtractor

router = APIRouter(prefix="/api", tags=["Explorer & Streaming"])

IGNORE_DIRS = {".git", ".indexes", "__pycache__", "venv", ".venv", "node_modules", "target", "build", "dist", ".idea", ".vscode", "scratch", "cloned_repos"}
IGNORE_EXTS = {".pyc", ".class", ".o", ".obj", ".exe", ".dll", ".so", ".bin", ".pkl", ".db", ".sqlite", ".index"}

LANG_EXT_MAP = {
    ".py": "Python",
    ".ipynb": "Jupyter Notebook",
    ".js": "JavaScript",
    ".ts": "TypeScript",
    ".html": "HTML",
    ".css": "CSS",
    ".md": "Markdown",
    ".toml": "TOML",
    ".yaml": "YAML",
    ".yml": "YAML",
    ".json": "JSON",
    ".sh": "Shell",
    ".java": "Java",
    ".go": "Go",
    ".rs": "Rust",
    ".cpp": "C++",
    ".c": "C",
}


def get_clean_files(rpath: Path) -> List[Path]:
    if not rpath or not rpath.exists():
        return []
    clean = []
    for f in rpath.glob("**/*"):
        if f.is_file():
            rel_parts = f.relative_to(rpath).parts
            if any(part in IGNORE_DIRS or part.startswith(".") for part in rel_parts):
                continue
            if f.suffix.lower() in IGNORE_EXTS:
                continue
            clean.append(f)
    return sorted(clean, key=lambda p: (p.parent.as_posix(), p.name))


def get_lang_stats(files: List[Path]) -> Dict[str, float]:
    counts = {}
    for f in files:
        lang = LANG_EXT_MAP.get(f.suffix.lower(), "Other")
        counts[lang] = counts.get(lang, 0) + 1
    total = sum(counts.values()) or 1
    return {lang: round((count / total) * 100, 1) for lang, count in sorted(counts.items(), key=lambda x: -x[1])}


def render_ipynb_to_markdown(raw_json: str) -> str:
    try:
        nb = json.loads(raw_json)
        md_parts = []
        for idx, cell in enumerate(nb.get("cells", []), 1):
            ctype = cell.get("cell_type", "code")
            source = "".join(cell.get("source", []))
            if ctype == "markdown":
                md_parts.append(f"{source}\n")
            elif ctype == "code":
                md_parts.append(f"```python\n# [In {idx}]\n{source}\n```")
                outputs = cell.get("outputs", [])
                out_texts = []
                for out in outputs:
                    if "text" in out:
                        out_texts.append("".join(out["text"]))
                    elif "data" in out and "text/plain" in out["data"]:
                        out_texts.append("".join(out["data"]["text/plain"]))
                if out_texts:
                    md_parts.append(f"```text\n# Output:\n{''.join(out_texts)[:500]}\n```")
        return "\n\n".join(md_parts)
    except Exception:
        return f"```json\n{raw_json}\n```"


class StreamChatRequest(BaseModel):
    query: str
    thread_id: Optional[str] = None


class StreamFileCopilotRequest(BaseModel):
    file_path: str
    query: str


class StreamCommitCopilotRequest(BaseModel):
    commit_sha: str


@router.get("/stats")
def get_repository_stats():
    """Returns real repository files, language distribution, and commit metadata."""
    from src.api.main import app_state

    rpath = app_state.active_repo_path
    if not rpath or not rpath.exists():
        rpath = Path.cwd()
        app_state.active_repo_path = rpath
        if not app_state.active_repo_source:
            app_state.active_repo_source = "https://github.com/sohamsabhaya/RepoLens"

    clean_files = get_clean_files(rpath)
    lang_stats = get_lang_stats(clean_files)
    detailed_commits = GitHistoryExtractor.get_detailed_commit_records(rpath, max_commits=30)
    
    total_lines = 0
    file_records = []
    for f in clean_files:
        rel = f.relative_to(rpath).as_posix()
        ext = f.suffix.lower()
        size_kb = round(f.stat().st_size / 1024.0, 1)
        lang = LANG_EXT_MAP.get(ext, "Other")
        
        lines = 0
        if ext not in [".png", ".jpg", ".jpeg", ".ico", ".pdf"]:
            try:
                lines = len(f.read_text(encoding="utf-8", errors="ignore").splitlines())
                total_lines += lines
            except Exception:
                pass

        file_records.append({
            "path": rel,
            "name": f.name,
            "ext": ext,
            "lang": lang,
            "size_kb": size_kb,
            "lines": lines,
        })

    readme_candidates = [f for f in clean_files if f.name.lower() in ["readme.md", "readme.markdown", "readme", "readme.txt", "readme.rst"]]
    has_readme = bool(readme_candidates)
    readme_content = ""
    readme_name = ""
    if has_readme:
        readme_path = readme_candidates[0]
        readme_name = readme_path.name
        try:
            readme_content = readme_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            pass

    return {
        "repo_source": app_state.active_repo_source or str(rpath),
        "repo_path": str(rpath),
        "total_files": len(clean_files),
        "total_commits": len(detailed_commits),
        "total_lines": total_lines,
        "primary_language": list(lang_stats.keys())[0] if lang_stats else "N/A",
        "languages": lang_stats,
        "files": file_records,
        "commits": detailed_commits,
        "has_readme": has_readme,
        "readme_name": readme_name,
        "readme_content": readme_content,
    }


@router.get("/file")
def get_file_details(path: str = Query(..., description="Relative file path")):
    """Returns raw and rendered content of a file."""
    from src.api.main import app_state
    rpath = app_state.active_repo_path
    if not rpath or not rpath.exists():
        raise HTTPException(status_code=400, detail="Repository not initialized.")

    target = (rpath / path).resolve()
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail=f"File '{path}' not found.")

    ext = target.suffix.lower()
    size_kb = round(target.stat().st_size / 1024.0, 1)

    if ext in [".png", ".jpg", ".jpeg", ".ico"]:
        return {
            "path": path,
            "is_image": True,
            "size_kb": size_kb,
            "ext": ext,
        }

    try:
        raw_text = target.read_text(encoding="utf-8", errors="ignore")
        lines = len(raw_text.splitlines())
        rendered_md = render_ipynb_to_markdown(raw_text) if ext == ".ipynb" else None
        
        return {
            "path": path,
            "is_image": False,
            "size_kb": size_kb,
            "ext": ext,
            "lang": LANG_EXT_MAP.get(ext, "text"),
            "lines": lines,
            "content": raw_text,
            "rendered_md": rendered_md,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read file: {e}")


@router.post("/stream/chat")
async def stream_chat(req: StreamChatRequest):
    """Streams real-time LLM tokens for full codebase Q&A."""
    from src.api.main import app_state

    llm = get_reasoning_llm()
    rag_context = ""
    citations = []

    if app_state.vectorstore:
        relevant_docs = app_state.vectorstore.similarity_search(req.query, k=5)
        chunks = []
        for d in relevant_docs:
            src = d.metadata.get("source_file") or d.metadata.get("file_path") or "code"
            lines = f"L{d.metadata.get('start_line', 1)}-L{d.metadata.get('end_line', 1)}"
            cit = f"[{src}:{lines}]"
            if cit not in citations:
                citations.append(cit)
            chunks.append(f"--- Document from {cit} ---\n{d.page_content}")
        rag_context = "\n\n".join(chunks)

    prompt = f"""You are RepoLens Codebase Intelligence Copilot.
Answer the user's question accurately using the following retrieved codebase context.

User Question: "{req.query}"

Retrieved Codebase Context:
```
{rag_context}
```

Provide a technical, complete, grounded response with exact citations (`[file.py:L10-L40]`)."""

    async def token_generator():
        # First send citation metadata header line if citations exist
        if citations:
            meta_json = json.dumps({"citations": citations})
            yield f"__CITATIONS__{meta_json}__END_CITATIONS__\n"
        for chunk in llm.stream(prompt):
            if chunk.content:
                yield chunk.content

    return StreamingResponse(token_generator(), media_type="text/plain; charset=utf-8")


@router.post("/stream/overview")
async def stream_overview():
    """Streams real-time AI architectural synthesis for the repository."""
    from src.api.main import app_state

    rpath = app_state.active_repo_path or Path.cwd()
    clean_files = get_clean_files(rpath)
    lang_stats = get_lang_stats(clean_files)

    sample_snippets = []
    key_names = ["main.py", "app.py", "server.py", "index.py", "package.json", "pyproject.toml", "requirements.txt", "dockerfile"]
    for f in clean_files:
        if f.name.lower() in key_names:
            try:
                sample_snippets.append(f"--- File: {f.name} ---\n{f.read_text(encoding='utf-8', errors='ignore')[:400]}\n")
            except Exception:
                pass
        if len(sample_snippets) >= 4:
            break

    file_summary_list = "\n".join([f"- {f.relative_to(rpath).as_posix()}" for f in clean_files[:35]])
    snippets_block = "\n".join(sample_snippets)

    prompt = f"""You are an expert Lead Software Architect and Principal Engineer analyzing a software repository.
Repository Source: {app_state.active_repo_source or 'Current Project'}
Primary Languages: {', '.join([f'{k} ({v:.1f}%)' for k, v in list(lang_stats.items())[:3]]) if lang_stats else 'Python'}
Total Files: {len(clean_files)}

Key Files in Codebase:
{file_summary_list}

Key Entrypoint / Config Snippets:
{snippets_block}

Generate a comprehensive, beautifully structured Architecture Overview in Markdown with the following sections:
1. 🎯 **Project Purpose & Core Goal** (What does this software do? Who is it for?)
2. 🏗️ **High-Level System Architecture & Flow** (Provide an ASCII or Mermaid diagram explaining how data/requests flow)
3. 📦 **Core Modules & Key Responsibilities** (Breakdown of packages and critical files)
4. ⚙️ **Tech Stack & Key Frameworks** (Languages, libraries, databases, APIs)
5. 🚀 **Setup, Execution & Entrypoints** (How to run, test, and deploy this project)

Tone: Highly professional, developer-friendly, and authoritative."""

    llm = get_reasoning_llm()

    async def token_generator():
        for chunk in llm.stream(prompt):
            if chunk.content:
                yield chunk.content

    return StreamingResponse(token_generator(), media_type="text/plain; charset=utf-8")


@router.post("/stream/file-copilot")
async def stream_file_copilot(req: StreamFileCopilotRequest):
    """Streams in-place RAG answers for a specific opened file."""
    from src.api.main import app_state

    rpath = app_state.active_repo_path or Path.cwd()
    target_file = rpath / req.file_path
    if not target_file.exists():
        raise HTTPException(status_code=404, detail=f"File {req.file_path} not found.")

    file_text = target_file.read_text(encoding="utf-8", errors="ignore")[:7000]
    rag_context = ""
    citations = []

    if app_state.vectorstore:
        relevant_docs = app_state.vectorstore.similarity_search(f"{req.file_path} {req.query}", k=4)
        chunks = []
        for d in relevant_docs:
            src = d.metadata.get("source_file") or d.metadata.get("file_path") or "code"
            lines = f"L{d.metadata.get('start_line', 1)}-L{d.metadata.get('end_line', 1)}"
            cit = f"[{src}:{lines}]"
            if cit not in citations:
                citations.append(cit)
            chunks.append(f"--- Chunk from {cit} ---\n{d.page_content[:600]}")
        rag_context = "\n\n".join(chunks)

    prompt = f"""You are RepoLens Code Copilot.
The user is asking a question specifically about file `{req.file_path}`.

User Query: "{req.query}"

Target File Content (`{req.file_path}`):
```
{file_text}
```

Related AST Chunks & Vector Context from Codebase:
```
{rag_context}
```

Provide a technical, structured, concise explanation. Cite specific functions and line ranges."""

    llm = get_reasoning_llm()

    async def token_generator():
        if citations:
            meta_json = json.dumps({"citations": citations})
            yield f"__CITATIONS__{meta_json}__END_CITATIONS__\n"
        for chunk in llm.stream(prompt):
            if chunk.content:
                yield chunk.content

    return StreamingResponse(token_generator(), media_type="text/plain; charset=utf-8")


@router.post("/stream/commit-copilot")
async def stream_commit_copilot(req: StreamCommitCopilotRequest):
    """Streams in-place AI explanations for a specific Git commit."""
    from src.api.main import app_state

    rpath = app_state.active_repo_path or Path.cwd()
    detailed_commits = GitHistoryExtractor.get_detailed_commit_records(rpath, max_commits=35)
    
    target_commit = next((c for c in detailed_commits if c["sha"] == req.commit_sha or c["full_sha"].startswith(req.commit_sha)), None)
    if not target_commit:
        raise HTTPException(status_code=404, detail=f"Commit {req.commit_sha} not found.")

    diff_summary = "\n".join([f"--- {d['file']} ---\n{d['patch'][:1000]}" for d in target_commit["diffs"]])
    prompt = f"""You are RepoLens Git Intelligence Copilot.
Explain the following Git Commit in technical detail:

Commit SHA: {target_commit['sha']}
Author: {target_commit['author']}
Message: {target_commit['message']}
Files Changed: {list(target_commit['files_stats'].keys())}

Diff Patches:
```diff
{diff_summary[:4000]}
```

Summarize what changed, why this change was made, and the architectural impact."""

    llm = get_reasoning_llm()

    async def token_generator():
        for chunk in llm.stream(prompt):
            if chunk.content:
                yield chunk.content

    return StreamingResponse(token_generator(), media_type="text/plain; charset=utf-8")
