# RepoLens — Autonomous AI Codebase Intelligence & Git Evolution Studio

<div align="center">

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?style=for-the-badge&logo=FastAPI&logoColor=white)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-FF6F00.svg?style=for-the-badge&logo=LangChain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![Groq](https://img.shields.io/badge/Groq-Llama_3.3_%26_Qwen-F55036.svg?style=for-the-badge)](https://groq.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16%20%2F%2018-4169E1.svg?style=for-the-badge&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB.svg?style=for-the-badge&logo=Python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

**An enterprise-grade, multi-turn AI codebase intelligence studio built with LangGraph, FastAPI, and a light-themed Web UI that parses multi-language AST syntax trees, indexes Git commit & diff evolution, performs CampusX Corrective RAG (CRAG) with confidence thresholds, and enables Human-in-the-Loop query disambiguation and state time-travel.**

[Key Features](#-key-features) • [Architecture & Workflow](#-architecture--workflow) • [Interfaces](#-dual-user-interfaces) • [Getting Started](#-getting-started) • [API Reference](#-api-reference) • [Project Structure](#-project-structure)

</div>

---

<div align="center">
  <img src="static/banner.png" alt="RepoLens Architecture and Codebase Exploration Engine" width="900" />
</div>

---

## 💡 Overview

Understanding unfamiliar, complex, or legacy codebases is challenging when relying solely on keyword search or flat documentation. Developers need to know not only **how** code functions today, but also **why** architectural decisions were made, **when** changes landed across Git commit lineages, and **how** project modules interact — with strict mathematical grounding and zero LLM hallucinations.

**RepoLens** solves this by unifying:
1. **Multi-Language AST Code Parsing**: Syntax-aware symbol extraction for functions, classes, docstrings, and line ranges across Python, TypeScript, JavaScript, Java, Go, Rust, and C++.
2. **Git Evolution & Diff Lineage Ingestion**: Indexes commit authors, commit messages, and structured unified diff patches to answer historical evolution questions.
3. **Deterministic Structural Tools**: Traverses physical directory trees and builds Git commit DAGs with zero hallucinations.
4. **CampusX Corrective RAG (CRAG)**: Evaluates document relevance using strict confidence thresholds ($\ge 0.7$, $0.3 - 0.7$, $< 0.3$) and knowledge strip decomposition.
5. **Human-in-the-Loop (HITL) & Time-Travel**: Pauses execution using LangGraph's native `interrupt()` on ambiguous queries, allowing human disambiguation and conversation state rollback.
6. **Enterprise Session Persistence**: Backed by PostgreSQL (`PostgresSaver`) and fast in-memory checkpointing (`MemorySaver`) with UUID thread isolation.

---

## ✨ Key Features

- 🧠 **Multi-Language AST Extraction**: Syntax-aware chunking for Python, Java, JavaScript, TypeScript, Go, Rust, and C++.
- 📜 **Git Commit & Diff Ingestion**: Ingests repository commits and diff patches to explain *"Why was this function changed?"*.
- 🌲 **Zero-Hallucination Structural Lookups**: Direct deterministic file trees and parent-child Git commit DAGs without LLM hallucinations.
- ⚖️ **CampusX Corrective RAG (CRAG)**:
  - **$\ge 0.7$ (High Confidence)**: Grounded code generation with exact `[file.py:L10-L40]` citations.
  - **$0.3 - 0.7$ (Ambiguous)**: Automatic knowledge strip refinement and query rewriting.
  - **$< 0.3$ (Irrelevant)**: Fallback search or graceful rejection.
- 🛑 **Human-in-the-Loop (HITL) Disambiguation**: Halts execution via `interrupt()` when multiple candidate files match a query, prompting the user for selection before resuming with `Command(resume=...)`.
- ⏳ **Time-Travel & Checkpoint Inspection**: Rewind conversations to any past state checkpoint and fork alternative reasoning paths.
- 🎨 **Modern Light-Themed Web UI & Streamlit Studio**: Clean single-page application with real-time token streaming, in-place RAG file copilot, and syntax-highlighted git diff explorer.

---

## 🏗️ Architecture & Workflow

RepoLens models the codebase intelligence, retrieval, and reasoning process as a stateful **LangGraph StateGraph**:

```mermaid
flowchart TD
    Start(["Start / User Query"]) --> route_query["route_query: Classify Query Intent<br/>Code RAG vs Git History vs Structure Tree"]
    
    route_query -->|"intent: tree"| get_file_tree["get_file_tree: Deterministic Zero-Hallucination<br/>Directory & Symbol Tree"]
    route_query -->|"intent: commit_history"| get_git_history["get_git_history: Extract Commit DAG,<br/>Authors & Diff Patches"]
    route_query -->|"intent: code_rag"| retrieve_docs["retrieve_docs: Dense Vector Retrieval<br/>AST Chunks & Symbol Metadata"]
    
    retrieve_docs --> grade_documents["grade_documents: CampusX CRAG Evaluator<br/>Score Relevance Confidence"]
    
    grade_documents -->|"confidence >= 0.7"| knowledge_filter["knowledge_filter: Decompose & Filter<br/>Relevant Knowledge Strips"]
    grade_documents -->|"0.3 <= confidence < 0.7"| detect_ambiguity["detect_ambiguity: Evaluate Symbol Ambiguity"]
    grade_documents -->|"confidence < 0.3"| web_search_fallback["web_search_fallback: External Docs<br/>& Web Search Fallback"]
    
    detect_ambiguity -->|"ambiguous candidates"| hitl_interrupt["hitl_interrupt: LangGraph interrupt()<br/>Prompt User Disambiguation"]
    detect_ambiguity -->|"single match"| query_rewrite["query_rewrite: Reformulate Query<br/>& Expand Synonyms"]
    
    query_rewrite --> retrieve_docs
    hitl_interrupt -->|"Command resume"| knowledge_filter
    
    knowledge_filter --> generate_answer["generate_answer: Groq Llama 3.3 Reasoning<br/>Synthesize Grounded Answer & Citations"]
    get_file_tree --> format_response["format_response: Format Tree / DAG Output"]
    get_git_history --> generate_answer
    web_search_fallback --> generate_answer
    
    generate_answer --> End(["End / Return Streamed Answer & Citations"])
    format_response --> End
```

### Graph Nodes Explained:

1. **`route_query`**: Analyzes the user query using structured Pydantic schemas to classify intent between *Code RAG*, *Git History*, or *Structural Tree*.
2. **`retrieve_docs`**: Executes dense vector similarity search (FAISS/Chroma) against AST code chunks and commit metadata.
3. **`grade_documents`**: Evaluates document relevance using the CampusX CRAG framework to compute a confidence score.
4. **`detect_ambiguity`**: Identifies whether the query references multiple conflicting symbols across different modules.
5. **`hitl_interrupt`**: Calls LangGraph's native `interrupt()` breakpoint to pause execution and prompt the developer for disambiguation.
6. **`query_rewrite`**: Reformulates under-specified or ambiguous queries for a secondary retrieval pass.
7. **`knowledge_filter`**: Deconstructs chunks into fine-grained knowledge strips, filtering out noise and non-relevant lines.
8. **`web_search_fallback`**: Triggers DuckDuckGo / external documentation search when internal code confidence is $< 0.3$.
9. **`generate_answer`**: Synthesizes a structured response grounded with verified inline file-line citations (`[file.py:L10-L40]`).
10. **`format_response`**: Formats deterministic tree and Git DAG structures with zero hallucination.

---

### ⚖️ Corrective RAG (CRAG) Decision Flow

```mermaid
flowchart TD
    RetrievedContext(["Retrieved AST Chunks"]) --> EvaluatorNode["evaluate_retrieval: Calculate Confidence Score"]
    
    EvaluatorNode --> DecisionNode{"Confidence Grade"}
    
    DecisionNode -->|"Score >= 0.7 (Correct)"| DirectSynthesis["High Confidence Pass<br/>Extract Knowledge Strips"]
    DecisionNode -->|"0.3 <= Score < 0.7 (Ambiguous)"| AmbiguityFlow["Refinement Pass<br/>Query Rewrite & HITL Check"]
    DecisionNode -->|"Score < 0.3 (Incorrect)"| FallbackFlow["Web Search Fallback<br/>DuckDuckGo / DDGS Search"]
    
    DirectSynthesis --> GroundedGen["Grounded Answer Generator"]
    AmbiguityFlow --> GroundedGen
    FallbackFlow --> GroundedGen
    
    GroundedGen --> FinalAnswer(["Verified Answer with Citations"])
```

---

### ⏳ State Persistence & Time-Travel History

```mermaid
flowchart LR
    Turn1["Checkpoint 1<br/>Thread UUID"] --> Turn2["Checkpoint 2<br/>AST Ingested"]
    Turn2 --> Turn3["Checkpoint 3<br/>Query 1"]
    Turn3 --> Turn4["Checkpoint 4<br/>Query 2"]
    Turn4 -.->|"Time-Travel Rewind"| Turn2
    Turn2 --> ForkedTurn["Forked Branch<br/>Alternative Reasoning Path"]
```

---

## 💻 Dual User Interfaces

### 1. Light-Themed Web Studio (`http://localhost:8000/`)
- Minimalist, distraction-free light theme with warm off-white tones (`#FAFAF8`) and IBM Plex typography.
- Real-time token streaming (`ReadableStream` text decoder).
- **Tab 1 (AI Code Copilot)**: Multi-turn chat with clickable vector citations (`[src/agent/models.py:L10-L40]`).
- **Tab 2 (Project Overview & README)**: Autonomous real-time AI architectural synthesis for repositories without a README, plus original README viewer.
- **Tab 3 (Codebase Explorer)**: Searchable file tree, Jupyter `.ipynb` rendered cells, image preview, and in-place Vector RAG File Copilot.
- **Tab 4 (Git Evolution & Diff Studio)**: Clickable commit cards, author metadata, in-place AI commit explainer, and expandable syntax-highlighted diffs (`+ green`, `- red`).

### 2. Streamlit Intelligence Studio (`http://localhost:8501/`)
- Multi-tab exploration studio with top repository metrics bar and ChatGPT-style sidebar conversation threads.

---

## 📁 Project Structure

```
RepoLens/
├── src/
│   ├── agent/                 # LangGraph workflows, HITL, models & checkpoints
│   │   ├── checkpoint.py      # UUID thread manager, MemorySaver & PostgresSaver
│   │   ├── hitl.py            # Ambiguity detector & LangGraph interrupt() nodes
│   │   ├── main_graph.py      # Compiled LangGraph state machine & router
│   │   ├── models.py          # Groq model factory (Reasoning & Fast router)
│   │   ├── router.py          # Intent classification schema & prompt
│   │   └── state.py           # Typed AgentState & Citation models
│   ├── api/                   # Production Modular FastAPI Backend
│   │   ├── main.py            # FastAPI app initialization, middleware & static mounting
│   │   ├── schemas.py         # Pydantic request & response schemas
│   │   └── routes/            # Route modules (chat, ingestion, structure, session, explorer)
│   │       ├── chat.py        # POST /api/chat, POST /api/resume
│   │       ├── ingestion.py   # POST /api/ingest
│   │       ├── structure.py   # GET /api/tree, GET /api/commits
│   │       ├── session.py     # GET /api/threads, GET /api/history/{id}
│   │       └── explorer.py    # GET /api/stats, GET /api/file, POST /api/stream/*
│   ├── ingestion/             # Codebase & Git parsing pipelines
│   │   ├── code_parser.py     # Multi-language AST chunker (Tree-Sitter)
│   │   ├── git_parser.py      # Git log, diff hunks & commit extractor
│   │   ├── tree_indexer.py    # Zero-hallucination file tree & commit DAG
│   │   └── pipeline.py        # Parallel multi-core ingestion runner
│   ├── rag/                   # Dense vector store & Corrective RAG
│   │   ├── crag.py            # CampusX CRAG thresholds & knowledge strips
│   │   ├── embeddings.py      # HuggingFace sentence-transformers
│   │   └── vectorstore.py     # Local FAISS indexing & metadata persistence
│   └── config.py              # Centralized environment & model configs
├── static/                    # Light-Themed Web UI & Static Assets
│   ├── index.html             # Single Page Application
│   └── banner.png             # Architecture Engine Banner
├── notebooks/                 # 100% Self-Contained Independent Stage Notebooks
│   ├── 01_codebase_agent_e2e.ipynb  # End-to-End Baseline RAG Orchestrator
│   ├── 02_git_history.ipynb         # Git History & Intent Diff RAG
│   ├── 03_crag_agent.ipynb          # CampusX Corrective RAG (CRAG)
│   └── 04_hitl_timetravel.ipynb     # LangGraph interrupt() & Time-Travel
├── tests/                     # Automated Unit & Integration Test Suite
│   ├── test_ingestion.py      # AST & parallel scan tests
│   ├── test_tree_dag.py       # Deterministic file tree & DAG tests
│   ├── test_router_and_checkpoints.py # UUID threads & Postgres fallback tests
│   ├── test_crag_eval.py      # CRAG threshold evaluation tests
│   └── run_all_tests.py       # Master test runner (9/9 passing)
├── docker-compose.yml         # PostgreSQL persistence service
├── streamlit_app.py           # Multi-tab Streamlit Web Application
├── requirements.txt           # Python dependencies
├── .env.example               # Environment variables template
└── README.md
```

---

## ⚡ Getting Started

### 1. Prerequisites & Installation

```bash
# Clone the repository
git clone https://github.com/sohamsabhaya/RepoLens.git
cd RepoLens

# Create and activate virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration

Create your `.env` file:
```bash
cp .env.example .env
```
Configure your keys in `.env`:
```ini
GROQ_API_KEY=gsk_your_groq_api_key_here
PRIMARY_LLM_MODEL=openai/gpt-oss-120b
FAST_ROUTER_MODEL=openai/gpt-oss-20b
DEFAULT_EMBEDDING_MODEL=all-MiniLM-L6-v2
POSTGRES_URL=postgresql://postgres:soham@localhost:5432/postgres
```

### 3. Launch Web Application

**Option A: Modern Light-Themed Web UI & FastAPI Server**
```bash
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```
Open **`http://localhost:8000/`** in your browser.

**Option B: Streamlit Intelligence Studio**
```bash
streamlit run streamlit_app.py
```
Open **`http://localhost:8501/`** in your browser.

---

## 🌐 API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the Light-Themed Web UI SPA. |
| `POST` | `/api/ingest` | Ingests repository from GitHub URL or local path into FAISS. |
| `GET` | `/api/stats` | Returns real-time file counts, lines, languages, and commits. |
| `GET` | `/api/file` | Inspects raw or rendered notebook/markdown file contents. |
| `POST` | `/api/stream/chat` | Streams real-time tokens for codebase Q&A with vector citations. |
| `POST` | `/api/stream/overview` | Streams real-time AI architectural synthesis for the repository. |
| `POST` | `/api/stream/file-copilot` | Streams in-place RAG answers for a specific opened file. |
| `POST` | `/api/stream/commit-copilot` | Streams in-place AI explanations for a specific Git commit. |
| `POST` | `/api/chat` | Multi-turn conversational query; yields pause status if HITL interrupt triggers. |
| `POST` | `/api/resume` | Resumes an interrupted thread with user's selected candidate. |
| `GET` | `/api/tree` | Returns deterministic zero-hallucination directory tree. |
| `GET` | `/api/commits` | Returns Git commit DAG with parent-child commit history. |
| `GET` | `/api/threads` | Lists active UUID thread IDs. |
| `GET` | `/api/history/{thread_id}` | Retrieves state checkpoints for time-travel inspection. |

---

## 🧪 Automated Tests

Run the complete 9-stage test suite:
```bash
python tests/run_all_tests.py
```
```
test_01_python_ast ... ok
test_02_java_chunking ... ok
test_03_parallel_scan ... ok
test_04_file_tree ... ok
test_05_commit_dag ... ok
test_06_uuid_threads ... ok
test_07_checkpointer ... ok
test_08_router_schema ... ok
test_09_crag_thresholds ... ok

----------------------------------------------------------------------
Ran 9 tests in 2.28s - OK (100% Pass Rate)
```

---

## 📄 License

Distributed under the **MIT License**. See `LICENSE` for more information.
