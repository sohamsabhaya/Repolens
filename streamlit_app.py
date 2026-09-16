"""
RepoLens — Autonomous AI Codebase Intelligence & Git Evolution Studio
Interactive File Explorer, In-Place RAG File Copilot, Jupyter Renderer, Real-time Streaming, and Visual Language Bar.
"""

import os
import re
import json
import uuid
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Generator
import streamlit as st
import dotenv

# Load environment
dotenv.load_dotenv()

from src.agent.models import get_reasoning_llm, get_embeddings_model
from src.agent.checkpoint import get_postgres_checkpointer, create_thread_config, generate_thread_id
from src.rag.vectorstore import load_or_build_vectorstore
from src.agent.main_graph import create_code_agent
from src.ingestion.git_parser import GitHistoryExtractor

# Page Configuration
st.set_page_config(
    page_title="RepoLens — AI Codebase Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    * {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    code, pre, .mono {
        font-family: 'JetBrains Mono', monospace !important;
    }

    .main-hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(135deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }

    .hero-subtitle {
        color: #94A3B8;
        font-size: 0.95rem;
        margin-bottom: 1.0rem;
    }

    /* Metric Tiles */
    .metric-card {
        background: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 10px;
        padding: 12px 16px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.35rem;
        font-weight: 700;
        color: #F8FAFC;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .ai-insight-box {
        background: #0F172A;
        border: 1px solid #38BDF8;
        border-radius: 10px;
        padding: 16px 20px;
        margin-top: 14px;
        margin-bottom: 14px;
    }

    .stat-add {
        color: #10B981;
        font-weight: 600;
    }
    .stat-del {
        color: #EF4444;
        font-weight: 600;
    }

    .badge-code {
        background-color: #0F172A;
        border: 1px solid #0284C7;
        color: #38BDF8;
        padding: 3px 8px;
        border-radius: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        display: inline-block;
        margin-right: 6px;
        margin-top: 4px;
    }
    .badge-git {
        background-color: #2E1065;
        border: 1px solid #9333EA;
        color: #E9D5FF;
        padding: 3px 8px;
        border-radius: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        display: inline-block;
        margin-right: 6px;
        margin-top: 4px;
    }

    /* Language Bar Colors */
    .lang-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 4px;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# Helpers
# ==========================================
IGNORE_DIRS = {".git", "__pycache__", "venv", ".venv", "node_modules", ".pytest_cache", ".indexes", "dist", "build"}
IGNORE_EXTS = {".pyc", ".lock", ".svg", ".min.js", ".map", ".pack", ".idx", ".sample", ".rev"}

LANG_COLOR_MAP = {
    "Python": "#3572A5",
    "Jupyter Notebook": "#DA5B0B",
    "JavaScript": "#F1E05A",
    "TypeScript": "#3178C6",
    "HTML": "#E34C26",
    "CSS": "#563D7C",
    "Markdown": "#083FA1",
    "TOML": "#9C4221",
    "YAML": "#CB171E",
    "JSON": "#292929",
    "Shell": "#89E051",
    "Other": "#64748B",
}


def get_clean_repo_files(repo_path: Path) -> List[Path]:
    """Returns real repository source files, ignoring .git and binary caches."""
    if not repo_path or not repo_path.exists():
        return []
    clean_files = []
    for f in repo_path.glob("**/*"):
        if f.is_file():
            if any(part in IGNORE_DIRS or part.startswith(".") for part in f.parts):
                continue
            if f.suffix.lower() in IGNORE_EXTS:
                continue
            clean_files.append(f)
    return sorted(clean_files, key=lambda p: (p.parent.as_posix(), p.name))


def get_language_stats(files: List[Path]) -> Dict[str, float]:
    """Calculates language distribution percentage."""
    ext_map = {
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
    }
    counts = {}
    for f in files:
        lang = ext_map.get(f.suffix.lower(), "Other")
        counts[lang] = counts.get(lang, 0) + 1
    total = sum(counts.values()) or 1
    return {lang: (count / total) * 100 for lang, count in sorted(counts.items(), key=lambda x: -x[1])}


def render_visual_language_bar(lang_stats: Dict[str, float]):
    """Renders a real multi-colored horizontal GitHub-style progress bar."""
    if not lang_stats:
        return
    
    bar_html_parts = []
    legend_html_parts = []
    
    for lang, pct in lang_stats.items():
        color = LANG_COLOR_MAP.get(lang, "#64748B")
        bar_html_parts.append(f'<div style="width:{pct:.1f}%; background:{color}; height:100%;" title="{lang}: {pct:.1f}%"></div>')
        legend_html_parts.append(
            f'<span style="font-size:0.85rem; margin-right:16px; color:#E2E8F0; white-space:nowrap;">'
            f'<span class="lang-dot" style="background:{color};"></span>'
            f'<b>{lang}</b> <span style="color:#94A3B8;">{pct:.1f}%</span></span>'
        )

    full_bar = f'<div style="display:flex; width:100%; height:10px; border-radius:6px; overflow:hidden; margin-top:6px; margin-bottom:10px;">{"".join(bar_html_parts)}</div>'
    full_legend = f'<div style="display:flex; flex-wrap:wrap; gap:8px; margin-bottom:16px;">{"".join(legend_html_parts)}</div>'
    st.markdown(full_bar + full_legend, unsafe_allow_html=True)


def render_ipynb_content(content: str) -> str:
    """Converts Jupyter Notebook JSON into clean readable Markdown/Python blocks."""
    try:
        nb = json.loads(content)
        md_parts = []
        cells = nb.get("cells", [])
        for idx, cell in enumerate(cells, 1):
            cell_type = cell.get("cell_type", "code")
            source = "".join(cell.get("source", []))
            
            if cell_type == "markdown":
                md_parts.append(f"{source}\n")
            elif cell_type == "code":
                md_parts.append(f"```python\n# [In {idx}]\n{source}\n```")
                
                outputs = cell.get("outputs", [])
                out_texts = []
                for out in outputs:
                    if "text" in out:
                        out_texts.append("".join(out["text"]))
                    elif "data" in out and "text/plain" in out["data"]:
                        out_texts.append("".join(out["data"]["text/plain"]))
                if out_texts:
                    combined_out = "\n".join(out_texts)[:400]
                    md_parts.append(f"```text\n# Output:\n{combined_out}\n```")
        return "\n\n".join(md_parts)
    except Exception as e:
        return f"```json\n{content}\n```"


def stream_llm_text(prompt: str) -> Generator[str, None, None]:
    """Streams LLM tokens for smooth ChatGPT-like response generation."""
    llm = get_reasoning_llm()
    for chunk in llm.stream(prompt):
        if chunk.content:
            yield chunk.content


# ==========================================
# Session State Initialization
# ==========================================
if "checkpointer" not in st.session_state:
    st.session_state["checkpointer"] = get_postgres_checkpointer()

if "current_thread_id" not in st.session_state:
    st.session_state["current_thread_id"] = generate_thread_id()

if "sessions_meta" not in st.session_state:
    st.session_state["sessions_meta"] = {
        st.session_state["current_thread_id"]: {
            "title": "New Conversation",
            "messages": [],
        }
    }

if "active_vectorstore" not in st.session_state:
    st.session_state["active_vectorstore"] = None

if "active_repo_source" not in st.session_state:
    st.session_state["active_repo_source"] = "."

if "active_repo_path" not in st.session_state:
    st.session_state["active_repo_path"] = Path.cwd()

if "file_qa_response" not in st.session_state:
    st.session_state["file_qa_response"] = None

if "commit_qa_response" not in st.session_state:
    st.session_state["commit_qa_response"] = None

if "ai_generated_overview" not in st.session_state:
    st.session_state["ai_generated_overview"] = None


def start_new_chat():
    new_tid = generate_thread_id()
    st.session_state["current_thread_id"] = new_tid
    st.session_state["sessions_meta"][new_tid] = {
        "title": "New Conversation",
        "messages": [],
    }


# ==========================================
# Sidebar: ChatGPT-Style Conversations
# ==========================================
with st.sidebar:
    st.markdown("### ⚡ **RepoLens Studio**")
    
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        start_new_chat()
        st.rerun()

    st.markdown("---")
    st.markdown("#### 💬 **Chat History**")

    for tid, session_info in reversed(list(st.session_state["sessions_meta"].items())):
        is_active = (tid == st.session_state["current_thread_id"])
        icon = "👉" if is_active else "💬"
        title = session_info.get("title", "Conversation")
        
        btn_label = f"{icon} {title[:28]}"
        if len(title) > 28:
            btn_label += "..."

        if st.button(btn_label, key=f"session_btn_{tid}", use_container_width=True):
            st.session_state["current_thread_id"] = tid
            st.rerun()

    st.markdown("---")
    st.markdown("#### ⚙️ **System Persistence**")
    chk_type = type(st.session_state["checkpointer"]).__name__
    is_pg = (chk_type == "PostgresSaver")
    status_color = "#10B981" if is_pg else "#F59E0B"
    status_label = "PostgreSQL 18 (Port 5432)" if is_pg else "In-Memory Fallback"
    
    st.markdown(f"""
    <div style="font-size:0.85rem; color:#94A3B8;">
        <div><b>Checkpointer:</b> <span style="color:{status_color}; font-weight:600;">● {status_label}</span></div>
        <div style="margin-top:4px;"><b>Inference:</b> <span style="color:#38BDF8; font-weight:600;">Groq Llama 3.3 70B</span></div>
        <div style="margin-top:4px;"><b>Embeddings:</b> <span style="color:#A78BFA; font-weight:600;">all-MiniLM-L6-v2</span></div>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# Main Header & Ingestion Bar
# ==========================================
st.markdown('<p class="main-hero-title">⚡ RepoLens Codebase Intelligence</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Interactive AST Code Analysis, Git Commit Diff Studio & Autonomous Reasoning</p>', unsafe_allow_html=True)

with st.container():
    col_input, col_btn = st.columns([4, 1.2])
    with col_input:
        repo_url = st.text_input(
            "Repository URL or Local Path:",
            value=st.session_state["active_repo_source"],
            placeholder="e.g. https://github.com/psf/requests or https://github.com/fastapi/fastapi or .",
            label_visibility="collapsed",
        )
    with col_btn:
        ingest_clicked = st.button("🚀 Ingest & Index", use_container_width=True, type="primary")

    if ingest_clicked:
        with st.spinner(f"Cloning & parsing {repo_url}..."):
            try:
                embeddings = get_embeddings_model()
                vstore = load_or_build_vectorstore(
                    repo_source=repo_url,
                    embeddings=embeddings,
                    force_reindex=True,
                )
                st.session_state["active_vectorstore"] = vstore
                st.session_state["active_repo_source"] = repo_url
                st.session_state["ai_generated_overview"] = None

                if repo_url.startswith(("http://", "https://", "git@")):
                    repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
                    cloned_local = Path.cwd() / "cloned_repos" / repo_name
                    if not cloned_local.exists():
                        cloned_local = Path.cwd() / ".indexes" / "cloned_repos" / repo_name
                    st.session_state["active_repo_path"] = cloned_local
                else:
                    st.session_state["active_repo_path"] = Path(repo_url).resolve()

                st.success(f"✅ Ingested **{repo_url}** successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Ingestion failed: {e}")

# Compute repository data
rpath = st.session_state.get("active_repo_path")
clean_files = get_clean_repo_files(rpath) if rpath else []
lang_stats = get_language_stats(clean_files) if clean_files else {}
detailed_commits = GitHistoryExtractor.get_detailed_commit_records(rpath, max_commits=30) if rpath else []

# Real Project Metrics Bar
if rpath and rpath.exists():
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{len(clean_files)}</div><div class="metric-label">Source Files</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{len(detailed_commits)}</div><div class="metric-label">Git Commits</div></div>', unsafe_allow_html=True)
    with m3:
        primary_lang = list(lang_stats.keys())[0] if lang_stats else "N/A"
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#38BDF8;">{primary_lang}</div><div class="metric-label">Primary Language</div></div>', unsafe_allow_html=True)
    with m4:
        total_lines = sum(len(f.read_text(encoding="utf-8", errors="ignore").splitlines()) for f in clean_files if f.suffix.lower() not in [".png", ".jpg", ".jpeg", ".ico"])
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#10B981;">{total_lines:,}</div><div class="metric-label">Total Code Lines</div></div>', unsafe_allow_html=True)

st.markdown("---")


# ==========================================
# Main Navigation Tabs
# ==========================================
tab_chat, tab_readme, tab_tree, tab_diffs = st.tabs([
    "💬 AI Code Copilot",
    "📖 Project Overview & README",
    "🌲 Interactive Codebase Explorer",
    "📜 Git Evolution & Diff Studio",
])

current_tid = st.session_state["current_thread_id"]
current_session = st.session_state["sessions_meta"].setdefault(current_tid, {"title": "New Conversation", "messages": []})


# ------------------------------------------
# Tab 1: AI Code Copilot (Real-time Streaming)
# ------------------------------------------
with tab_chat:
    if not rpath or not rpath.exists():
        st.info("👈 Enter a repository above and click **'🚀 Ingest & Index'** to start asking questions!")
    else:
        if not current_session["messages"]:
            st.markdown("#### 💡 Suggested Questions to Explore:")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔍 Explain the core architecture and key modules", use_container_width=True):
                    st.session_state["prompt_shortcut"] = "Explain the core architecture and key modules of this codebase."
                    st.rerun()
                if st.button("🌲 What is the folder and file structure?", use_container_width=True):
                    st.session_state["prompt_shortcut"] = "What is the folder and file structure of this repository?"
                    st.rerun()
            with col2:
                if st.button("📜 Explain recent git commits and why changes were made", use_container_width=True):
                    st.session_state["prompt_shortcut"] = "Explain the recent git commits and why changes were made."
                    st.rerun()
                if st.button("🛡️ How does error handling or agent workflow work?", use_container_width=True):
                    st.session_state["prompt_shortcut"] = "How does the agent workflow and error handling work in this repo?"
                    st.rerun()

        # Display history
        for msg in current_session["messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("citations"):
                    st.markdown("**Citations:**")
                    for cit in msg["citations"]:
                        if "commit:" in cit.lower():
                            st.markdown(f'<span class="badge-git">{cit}</span>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<span class="badge-code">{cit}</span>', unsafe_allow_html=True)

        user_input = st.chat_input("Ask anything about functions, classes, commits, or structure...")
        if "prompt_shortcut" in st.session_state:
            user_input = st.session_state.pop("prompt_shortcut")

        if user_input:
            if current_session["title"] == "New Conversation":
                clean_title = user_input.strip().replace("\n", " ")
                current_session["title"] = clean_title[:32] if len(clean_title) <= 32 else clean_title[:29] + "..."

            current_session["messages"].append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            # Generate Agent Response with Real-Time Streaming
            with st.chat_message("assistant"):
                try:
                    if st.session_state["active_vectorstore"] is None:
                        embeddings = get_embeddings_model()
                        st.session_state["active_vectorstore"] = load_or_build_vectorstore(
                            repo_source=st.session_state["active_repo_source"],
                            embeddings=embeddings,
                            force_reindex=False,
                        )

                    vstore = st.session_state["active_vectorstore"]
                    relevant_docs = vstore.similarity_search(user_input, k=6)
                    citations = []
                    context_blocks = []
                    for d in relevant_docs:
                        src = d.metadata.get("source_file") or d.metadata.get("file_path") or "code"
                        lines = f"L{d.metadata.get('start_line', 1)}-L{d.metadata.get('end_line', 1)}"
                        cit = f"[{src}:{lines}]"
                        if cit not in citations:
                            citations.append(cit)
                        context_blocks.append(f"--- Document from {cit} ---\n{d.page_content}")

                    prompt = f"""You are RepoLens Codebase Intelligence Copilot.
Answer the user's question accurately using the following retrieved codebase context.

User Question: "{user_input}"

Retrieved Codebase Context:
```
{chr(10).join(context_blocks)}
```

Provide a technical, complete, grounded response with exact citations (`[file.py:L10-L40]`)."""

                    # Real-time Streaming Output!
                    response_text = st.write_stream(stream_llm_text(prompt))

                    if citations:
                        st.markdown("**Citations:**")
                        for cit in citations:
                            if "commit:" in cit.lower():
                                st.markdown(f'<span class="badge-git">{cit}</span>', unsafe_allow_html=True)
                            else:
                                st.markdown(f'<span class="badge-code">{cit}</span>', unsafe_allow_html=True)

                    current_session["messages"].append({
                        "role": "assistant",
                        "content": response_text,
                        "citations": citations,
                    })

                except Exception as e:
                    st.error(f"Reasoning Error: {e}")


# ------------------------------------------
# Tab 2: Project Overview & Rich README Viewer (With Auto AI Generation)
# ------------------------------------------
with tab_readme:
    if not rpath or not rpath.exists():
        st.info("Ingest a repository above to view its project overview and documentation.")
    else:
        st.markdown("#### 📊 **Repository Language Distribution**")
        render_visual_language_bar(lang_stats)

        readme_candidates = [f for f in clean_files if f.name.lower() in ["readme.md", "readme.markdown", "readme", "readme.txt", "readme.rst"]]
        has_readme = bool(readme_candidates)
        
        col_hdr, col_btn = st.columns([3.2, 1.8])
        with col_hdr:
            st.markdown("### 📖 **Project Documentation & Architecture Overview**")
        with col_btn:
            btn_label = "🔄 Regenerate AI Architecture Overview" if st.session_state.get("ai_generated_overview") else "🤖 Generate AI Architectural Overview"
            if st.button(btn_label, use_container_width=True):
                st.session_state["trigger_ai_overview"] = True
                st.session_state["ai_generated_overview"] = None

        # Build context if we need to generate/stream the overview
        should_generate = st.session_state.get("trigger_ai_overview") or (not has_readme and not st.session_state.get("ai_generated_overview"))

        if not has_readme:
            st.markdown("""
            <div style="background:rgba(56, 189, 248, 0.08); border:1px solid #38BDF8; border-radius:8px; padding:14px 18px; margin-top:8px; margin-bottom:16px;">
                <div style="color:#38BDF8; font-weight:700; font-size:0.98rem; margin-bottom:4px;">
                    💡 Autonomous Architecture Overview (No README Detected)
                </div>
                <div style="color:#94A3B8; font-size:0.87rem; line-height:1.4;">
                    No <code>README.md</code> was found in this repository. RepoLens has autonomously inspected the codebase AST, source modules, and dependencies to synthesize the comprehensive architectural overview below in real time.
                </div>
            </div>
            """, unsafe_allow_html=True)

        if should_generate:
            st.session_state.pop("trigger_ai_overview", None)
            
            # Extract sample snippets from key entrypoints if present
            sample_snippets = []
            key_names = ["main.py", "app.py", "server.py", "index.py", "package.json", "pyproject.toml", "requirements.txt", "dockerfile"]
            for f in clean_files:
                if f.name.lower() in key_names:
                    try:
                        content_sample = f.read_text(encoding="utf-8", errors="ignore")[:400]
                        sample_snippets.append(f"--- File: {f.name} ---\n{content_sample}\n")
                    except Exception:
                        pass
                if len(sample_snippets) >= 4:
                    break

            file_summary_list = "\n".join([f"- {f.relative_to(rpath).as_posix()}" for f in clean_files[:35]])
            snippets_block = "\n".join(sample_snippets)

            overview_prompt = f"""You are an expert Lead Software Architect and Principal Engineer analyzing a software repository.
Repository Source: {st.session_state['active_repo_source']}
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

            with st.container():
                st.markdown("""
                <div style="background:#0F172A; border:1px solid #38BDF8; border-radius:8px; padding:12px 16px; margin-bottom:14px;">
                    <span style="color:#38BDF8; font-weight:700;">⚡ Streaming Real-Time AI Architectural Synthesis:</span>
                </div>
                """, unsafe_allow_html=True)
                
                # Real-time token streaming
                st.session_state["ai_generated_overview"] = st.write_stream(stream_llm_text(overview_prompt))

        elif st.session_state.get("ai_generated_overview"):
            st.markdown("""
            <div style="background:#0F172A; border:1px solid #38BDF8; border-radius:8px; padding:12px 16px; margin-bottom:14px;">
                <span style="color:#38BDF8; font-weight:700;">🤖 AI-Synthesized Codebase Architecture Overview:</span>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(st.session_state["ai_generated_overview"], unsafe_allow_html=True)

        # If README exists, also render original README in a dedicated view
        if has_readme:
            readme_path = readme_candidates[0]
            st.markdown("---")
            with st.expander(f"📄 View Original Repository `{readme_path.name}`", expanded=(not bool(st.session_state.get("ai_generated_overview")))):
                try:
                    raw_readme = readme_path.read_text(encoding="utf-8", errors="ignore")
                    for img_match in re.findall(r'!\[.*?\]\((.*?)\)', raw_readme):
                        if not img_match.startswith(("http://", "https://")):
                            local_img = rpath / img_match
                            if local_img.exists():
                                raw_readme = raw_readme.replace(f"({img_match})", f"({local_img.as_uri()})")
                    st.markdown(raw_readme, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Could not render README: {e}")


# ------------------------------------------
# Tab 3: Interactive Codebase Explorer & In-Place RAG File Copilot
# ------------------------------------------
with tab_tree:
    if not rpath or not rpath.exists():
        st.info("Ingest a repository above to explore its codebase.")
    else:
        st.subheader("🌲 Interactive Codebase Explorer & File Studio")
        
        col_nav, col_view = st.columns([1.3, 2.2])
        
        with col_nav:
            st.markdown("#### 📂 **File Directory**")
            
            search_query = st.text_input("Filter files:", placeholder="e.g. auth, api, test, .py")
            
            rel_files = [f.relative_to(rpath).as_posix() for f in clean_files]
            if search_query:
                rel_files = [rf for rf in rel_files if search_query.lower() in rf.lower()]

            file_icons = {
                ".py": "🐍",
                ".ipynb": "📓",
                ".html": "🌐",
                ".js": "📜",
                ".ts": "🔷",
                ".md": "📄",
                ".toml": "⚙️",
                ".json": "📦",
                ".yml": "⚙️",
                ".yaml": "⚙️",
                ".png": "🖼️",
                ".jpg": "🖼️",
            }
            
            file_display_names = {
                rf: f"{file_icons.get(Path(rf).suffix.lower(), '📄')} {rf}" for rf in rel_files
            }

            selected_file_rel = st.selectbox(
                "Select file to open:",
                options=rel_files if rel_files else ["No files matching filter"],
                format_func=lambda x: file_display_names.get(x, x),
            )

            # In-Place AI File Copilot (Accordion Tool)
            if selected_file_rel and selected_file_rel != "No files matching filter":
                st.markdown("---")
                with st.expander(f"🤖 **Ask AI Copilot About `{Path(selected_file_rel).name}`**", expanded=True):
                    st.caption("Perform semantic Vector RAG queries specifically about this opened file.")
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("🔍 Explain This File", use_container_width=True):
                            st.session_state["file_qa_trigger"] = f"Explain the purpose, key functions, and architecture of `{selected_file_rel}` in this codebase."
                    with c2:
                        if st.button("🛡️ Review Bugs/Smells", use_container_width=True):
                            st.session_state["file_qa_trigger"] = f"Review `{selected_file_rel}` for potential bugs, security concerns, or architectural anti-patterns."

                    file_custom_q = st.text_input("Custom question about this file:", placeholder="e.g. How does method X handle errors?")
                    if st.button("⚡ Ask AI Copilot", use_container_width=True) and file_custom_q:
                        st.session_state["file_qa_trigger"] = f"In `{selected_file_rel}`: {file_custom_q}"

        with col_view:
            if selected_file_rel and selected_file_rel != "No files matching filter":
                target_file = rpath / selected_file_rel
                st.markdown(f"#### 📝 **Viewing:** `{selected_file_rel}`")
                
                # If an in-place File AI answer was requested, perform RAG retrieval and stream above code
                if "file_qa_trigger" in st.session_state:
                    trigger_q = st.session_state.pop("file_qa_trigger")
                    st.markdown(f"""
                    <div class="ai-insight-box">
                        <div style="font-size:0.92rem; font-weight:700; color:#38BDF8; margin-bottom:6px;">
                            🤖 RAG Copilot Insight for <code>{selected_file_rel}</code>:
                        </div>
                        <div style="font-size:0.85rem; color:#94A3B8; margin-bottom:10px;">
                            <i>"{trigger_q}"</i>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    try:
                        file_text = target_file.read_text(encoding="utf-8", errors="ignore")[:7000]
                        rag_context = ""
                        citations = []
                        if st.session_state.get("active_vectorstore"):
                            vstore = st.session_state["active_vectorstore"]
                            relevant_docs = vstore.similarity_search(f"{selected_file_rel} {trigger_q}", k=4)
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
The user is asking a question specifically about file `{selected_file_rel}`.

User Query: "{trigger_q}"

Target File Content (`{selected_file_rel}`):
```
{file_text}
```

Related AST Chunks & Vector Context from Codebase:
```
{rag_context}
```

Provide a technical, structured, concise explanation. Cite specific functions and line ranges."""
                        
                        # Real-time Streaming Output!
                        resp_content = st.write_stream(stream_llm_text(prompt))
                        st.session_state["file_qa_response"] = (trigger_q, resp_content, citations)
                        
                        if citations:
                            st.markdown("**Vector Sources Referenced:**")
                            for c in citations:
                                st.markdown(f'<span class="badge-code">{c}</span>', unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error querying RAG AI: {e}")
                    
                    st.markdown("</div>", unsafe_allow_html=True)

                elif st.session_state.get("file_qa_response"):
                    q_asked, resp_content, cits = st.session_state["file_qa_response"]
                    st.markdown(f"""
                    <div class="ai-insight-box">
                        <div style="font-size:0.92rem; font-weight:700; color:#38BDF8; margin-bottom:6px;">
                            🤖 RAG Copilot Insight for <code>{selected_file_rel}</code>:
                        </div>
                        <div style="font-size:0.85rem; color:#94A3B8; margin-bottom:10px;">
                            <i>"{q_asked}"</i>
                        </div>
                        <div style="font-size:0.92rem; color:#E2E8F0;">
                    """, unsafe_allow_html=True)
                    st.markdown(resp_content)
                    if cits:
                        st.markdown("**Vector Sources Referenced:**")
                        for c in cits:
                            st.markdown(f'<span class="badge-code">{c}</span>', unsafe_allow_html=True)
                    st.markdown("</div></div>", unsafe_allow_html=True)

                # Render File Content based on Type
                if target_file.exists() and target_file.is_file():
                    try:
                        ext = target_file.suffix.lower()
                        size_kb = target_file.stat().st_size / 1024.0

                        if ext in [".png", ".jpg", ".jpeg", ".ico"]:
                            st.caption(f"🖼️ Image File | Size: {size_kb:.1f} KB")
                            st.image(str(target_file))

                        elif ext == ".ipynb":
                            st.caption(f"📓 Jupyter Notebook | Size: {size_kb:.1f} KB")
                            raw_content = target_file.read_text(encoding="utf-8", errors="ignore")
                            nb_view_mode = st.radio("Notebook Mode:", ["Rendered Notebook", "Raw JSON"], horizontal=True)
                            if nb_view_mode == "Rendered Notebook":
                                formatted_nb = render_ipynb_content(raw_content)
                                st.markdown(formatted_nb, unsafe_allow_html=True)
                            else:
                                st.code(raw_content, language="json", line_numbers=True)

                        elif ext == ".md":
                            raw_content = target_file.read_text(encoding="utf-8", errors="ignore")
                            st.caption(f"📄 Markdown Document | Lines: {len(raw_content.splitlines())} | Size: {size_kb:.1f} KB")
                            md_mode = st.radio("Display Mode:", ["Formatted Preview", "Raw Source"], horizontal=True)
                            if md_mode == "Formatted Preview":
                                st.markdown(raw_content, unsafe_allow_html=True)
                            else:
                                st.code(raw_content, language="markdown", line_numbers=True)

                        else:
                            raw_content = target_file.read_text(encoding="utf-8", errors="ignore")
                            lines_cnt = len(raw_content.splitlines())
                            st.caption(f"📊 Lines: {lines_cnt} | Size: {size_kb:.1f} KB | Language: `{ext.lstrip('.') or 'text'}`")
                            lang_code = "python" if ext == ".py" else (ext.lstrip(".") or "text")
                            st.code(raw_content, language=lang_code, line_numbers=True)

                    except Exception as e:
                        st.error(f"Could not load file: {e}")


# ------------------------------------------
# Tab 4: Git Evolution & Diff Studio (Real-Time Streaming)
# ------------------------------------------
with tab_diffs:
    if not rpath or not rpath.exists():
        st.info("Ingest a repository above to view its Git history.")
    else:
        st.subheader("📜 Git Evolution & Interactive Commit Diff Studio")
        
        if not detailed_commits:
            st.info("No Git commit history available for this repository.")
        else:
            commit_options = [
                f"[{c['sha']}] {c['message'][:50]}... — {c['author']} (+{c['total_insertions']}/-{c['total_deletions']})"
                for c in detailed_commits
            ]
            
            selected_commit_idx = st.radio(
                "🔎 **Click on any commit below to open full details & diffs:**",
                range(len(commit_options)),
                format_func=lambda i: commit_options[i],
                horizontal=False,
            )
            
            sel = detailed_commits[selected_commit_idx]
            
            # Commit Inspection Card
            with st.container():
                st.markdown(f"""
                <div style="background:#0F172A; border:1px solid #38BDF8; border-radius:10px; padding:18px 22px; margin-top:14px; margin-bottom:16px;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                        <span style="font-size:1.1rem; font-weight:700; color:#38BDF8; font-family:'JetBrains Mono', monospace;">🏷️ Commit {sel['sha']}</span>
                        <span style="color:#94A3B8; font-size:0.85rem;">📅 {sel['date']}</span>
                    </div>
                    <div style="font-size:1.05rem; font-weight:600; color:#F8FAFC; margin-bottom:8px;">
                        💬 {sel['message']}
                    </div>
                    <div style="font-size:0.85rem; color:#94A3B8; display:flex; gap:20px; flex-wrap:wrap; margin-top:10px;">
                        <span>👤 <b>Author:</b> {sel['author']} &lt;{sel['email']}&gt;</span>
                        <span>📄 <b>Files Changed:</b> {len(sel['files_stats'])}</span>
                        <span><span class="stat-add">+{sel['total_insertions']}</span> / <span class="stat-del">-{sel['total_deletions']}</span> lines</span>
                        <span style="font-family:'JetBrains Mono', monospace; font-size:0.75rem; color:#64748B;">SHA: {sel['full_sha']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # In-Place Action: Stream AI Copilot explanation
                if st.button(f"🤖 Ask AI Copilot: Explain Commit {sel['sha']}", use_container_width=True):
                    st.markdown(f"""
                    <div class="ai-insight-box">
                        <div style="font-size:0.9rem; font-weight:700; color:#38BDF8; margin-bottom:6px;">
                            🤖 AI Analysis for Commit <code>{sel['sha']}</code>:
                        </div>
                    """, unsafe_allow_html=True)
                    
                    try:
                        diff_summary = "\n".join([f"--- {d['file']} ---\n{d['patch'][:1000]}" for d in sel["diffs"]])
                        prompt = f"""You are RepoLens Git Intelligence Copilot.
Explain the following Git Commit in technical detail:

Commit SHA: {sel['sha']}
Author: {sel['author']}
Message: {sel['message']}
Files Changed: {list(sel['files_stats'].keys())}

Diff Patches:
```diff
{diff_summary[:4000]}
```

Summarize what changed, why this change was made, and the architectural impact."""
                        
                        commit_ai_text = st.write_stream(stream_llm_text(prompt))
                        st.session_state["commit_qa_response"] = (sel['sha'], commit_ai_text)
                    except Exception as e:
                        st.error(f"Error explaining commit: {e}")
                    
                    st.markdown("</div>", unsafe_allow_html=True)

                elif st.session_state.get("commit_qa_response"):
                    sha_resp, resp_text = st.session_state["commit_qa_response"]
                    if sha_resp == sel['sha']:
                        st.markdown(f"""
                        <div class="ai-insight-box">
                            <div style="font-size:0.9rem; font-weight:700; color:#38BDF8; margin-bottom:6px;">
                                🤖 AI Analysis for Commit <code>{sha_resp}</code>:
                            </div>
                            <div style="font-size:0.92rem; color:#E2E8F0;">
                        """, unsafe_allow_html=True)
                        st.markdown(resp_text)
                        st.markdown("</div></div>", unsafe_allow_html=True)

                # Modified Files & Line-by-Line Diffs
                st.markdown("#### 📂 **Modified Files & Syntax-Highlighted Diffs:**")
                
                if sel["diffs"]:
                    for d_idx, d in enumerate(sel["diffs"]):
                        with st.expander(f"📄 `{d['file']}` (Change: **{d['change_type']}**)", expanded=(d_idx == 0)):
                            st.code(d["patch"], language="diff", line_numbers=True)
                else:
                    st.info("ℹ️ Initial root commit or merge without a direct parent diff.")
