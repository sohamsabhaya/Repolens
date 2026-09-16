# Codebase Intelligence Agent — Master Architecture & Phased SDLC Plan

**A 5-Stage Progressive Roadmap: From Baseline GitHub Code RAG to Advanced Multi-Stream Agentic Architecture with Time-Travel, Human-in-the-Loop, FastAPI, and Streamlit UI.**

---

## 1. Phased SDLC Overview & Git Checkpoints

We build this system incrementally in **5 disciplined, testable stages**. At the completion of each stage, all features are tested, validated, and saved with a dedicated Git commit.

```
┌────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: Baseline Code RAG Chatbot (GitHub Clone + AST + Linear RAG)  │
│ └── Git Commit: "feat(stage1): baseline github repo code rag chatbot" │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│ STAGE 2: Multi-Stream History Ingestion (Commits, Diffs & Summaries)   │
│ └── Git Commit: "feat(stage2): git commit history & diff intent rag"  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│ STAGE 3: Deterministic Structural Lookups & Query Router              │
│ └── Git Commit: "feat(stage3): deterministic file tree & commit dag"  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│ STAGE 4: Advanced Subgraphs, CRAG, Self-RAG, HITL & Time-Travel        │
│ └── Git Commit: "feat(stage4): crag self-rag subgraph hitl time-travel"│
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│ STAGE 5: Production Modularization, FastAPI Server & Streamlit UI      │
│ └── Git Commit: "feat(stage5): modular src layout fastapi streamlit ui"│
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Models & Core Stack (Groq-Only Architecture)

| Component | Technology | Provider | Purpose |
| :--- | :--- | :--- | :--- |
| **Primary LLM** | `openai/gpt-oss-120b` or `qwen/qwen3.8-27b` | **Groq** | Main reasoning, generation, Self-RAG grading |
| **Fast LLM** | `qwen/qwen3.8-27b` or `openai/gpt-oss-20b` | **Groq** | Query routing, CRAG doc grading, query refinement |
| **Embeddings** | `BAAI/bge-small-en-v1.5` / `all-MiniLM-L6-v2` | **Hugging Face** | Local dense vector embeddings (zero cost, 100% offline) |
| **Vector Store** | `FAISS` (`langchain_community.vectorstores.FAISS`) | **Local** | Fast similarity search with metadata filtering |
| **Graph Framework**| `LangGraph` | **LangChain** | Dual-graphs, subgraphs, checkpointers, HITL |
| **Observability** | `LangSmith` | **LangSmith** | Complete nested execution tracing |
| **Backend API** | `FastAPI` | **Uvicorn** | Async SSE token streaming, REST endpoints |
| **Frontend UI** | `Streamlit` | **Streamlit** | Collapsible file tree, chat with citations, commit DAG |

---

## 3. Detailed Breakdown of Each Stage

### 🚀 Stage 1: Baseline GitHub Repo Code RAG Chatbot
- **Goal**: Create a fully working baseline chatbot that clones any GitHub repo (or takes a local repo path), indexes source code, and chats about code using linear RAG + conversational memory.
- **What We Build in Notebook (`notebooks/01_codebase_agent_e2e.ipynb`)**:
  1. Setup `.env`, Groq LLM client, and local HuggingFace embeddings (`sentence-transformers`).
  2. Repo Cloner / Path Ingestor: accepts GitHub URL (e.g. `https://github.com/user/repo`) or local path.
  3. Python AST Code Chunker: extracts functions, classes, line numbers, and docstrings.
  4. Non-code fallback chunker: splits `README.md`, configs, and requirements.
  5. Local FAISS Vector Store: creates and persists the `code` index.
  6. Linear StateGraph: `retrieve -> generate` with `MemorySaver` / `SqliteSaver` checkpointer for multi-turn chat.
  7. Test with code questions and inline citations (`[file.py:L10-L40]`).
- **✅ Verification & Git Commit**: `feat(stage1): baseline github repo code rag chatbot`

---

### 📜 Stage 2: Git History, Commits & Intent Diff Ingestion
- **Goal**: Expand ingestion to Git commit history, file diffs, and intent summaries, enabling the chatbot to answer *"why/when did this change?"*.
- **What We Build**:
  1. Git Log Parser: extracts commit SHA, author, date, message, changed files (`--numstat`).
  2. Git Diff Extractor & Trimmer: extracts modified hunks with noise/lockfile filtering.
  3. LLM Diff Summarizer: generates 1-2 sentence natural-language intent summaries at ingestion time.
  4. Multi-Stream FAISS Index: creates a dedicated `commit` index alongside the `code` index.
  5. Dual-Stream Retriever: retrieves both code chunks and commit records for historical inquiries.
  6. Test with history-based questions (e.g., *"Why was function X changed?", "When was feature Y introduced?"*).
- **✅ Verification & Git Commit**: `feat(stage2): git commit history & diff intent rag`

---

### 🌳 Stage 3: Deterministic Structural Lookups & Scope Router
- **Goal**: Add deterministic JSON file tree and commit DAG lookups so structural questions are answered with zero LLM hallucination.
- **What We Build**:
  1. Deterministic JSON File Tree Builder: walks directory hierarchy and records paths, types, sizes.
  2. Deterministic Commit DAG Builder: parses `git log --all --graph --parents` into parent-child commit relations.
  3. `QuestionClassification` Router Node: LLM classifies queries into `code`, `commit`, `tree`, or `mixed`.
  4. `tree_tool` Node: executes deterministic data lookups directly without calling an LLM.
  5. `format_response` Node: formats tool outputs and RAG answers with clickable references.
  6. Test deterministic lookups (e.g., *"What files are in the auth directory?", "Show commit history for main.py"*).
- **✅ Verification & Git Commit**: `feat(stage3): deterministic file tree & commit dag`

---

### 🧠 Stage 4: Advanced Subgraphs, CRAG, Self-RAG, HITL & Time-Travel
- **Goal**: Convert RAG into an independent Subgraph, implement CRAG + Self-RAG quality loops, and add Human-in-the-Loop & Time-Travel.
- **What We Build**:
  1. **RAG Subgraph**: Encapsulates isolated `RagState` with `Annotated[List[GradedDoc], operator.add]` state reducers.
  2. **CRAG (Corrective RAG)**:
     - `grade_documents` node: strictly grades relevance of each retrieved chunk.
     - `refine_query` node: rewrites search query if docs are insufficient (capped at 2 retries).
  3. **Self-RAG (Generation Verification)**:
     - `grade_generation` node: verifies `grounded` (hallucination check) and `answers_question` (utility check).
  4. **Human-in-the-Loop (HITL)**:
     - Ambiguity detection: triggers `interrupt()` when a query matches multiple ambiguous functions/files.
     - Resumption: resumes execution via `Command(resume=...)` without restarting the graph.
  5. **Time-Travel & State History**:
     - Inspect past steps via `app.get_state_history(config)`.
     - Edit state and fork execution using `app.update_state()`.
- **✅ Verification & Git Commit**: `feat(stage4): crag self-rag subgraph hitl time-travel`

---

### 📦 Stage 5: Production Modularization, FastAPI Server & Streamlit UI
- **Goal**: Refactor the entire tested system into a clean `src/` package, build the FastAPI backend, and launch the multi-panel Streamlit UI.
- **What We Build**:
  1. Modular `src/` Layout:
     - `src/ingestion/` (`code_parser.py`, `git_parser.py`, `tree_indexer.py`, `pipeline.py`)
     - `src/rag/` (`embeddings.py`, `vectorstore.py`, `retriever.py`)
     - `src/agent/` (`state.py`, `prompts.py`, `models.py`, `checkpoint.py`, `main_graph.py`, `subgraphs/rag_subgraph.py`)
  2. FastAPI Service (`src/api/server.py`):
     - `POST /api/ingest` (Ingests repository from GitHub URL or local path)
     - `POST /api/chat` (SSE token streaming with citations)
     - `GET /api/tree` & `GET /api/commits` (Deterministic structures)
     - `POST /api/resume` (HITL disambiguation response)
  3. Streamlit Multi-Panel UI (`src/ui/app.py`):
     - **Left Panel**: Collapsible repository file tree.
     - **Center Panel**: Conversational chat with streaming markdown and citation badges.
     - **Right Panel**: Visual Git commit DAG graph (`st.graphviz_chart` / ASCII view).
- **✅ Verification & Git Commit**: `feat(stage5): modular src layout fastapi streamlit ui`

---

## 4. Immediate Action: Starting Stage 1

We will now start with **Stage 1**:
1. Initialize the `notebooks/` directory.
2. Create `notebooks/01_codebase_agent_e2e.ipynb`.
3. Implement the GitHub Cloner / Path Reader, AST Code Chunker, Local FAISS Store, and Linear RAG Chatbot.
