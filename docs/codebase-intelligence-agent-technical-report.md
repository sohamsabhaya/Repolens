# Codebase Intelligence Agent — Technical Architecture Report

This report elaborates the plan into a concrete, buildable design: graph structure, state schemas, prompts, free model choices, checkpointing, and UI — mirroring the style of your reference notebook (TypedDict + Pydantic state, `StateGraph` with conditional edges, clean system-prompt constants) but scoped for a **retrieval-and-explain agent**, not a writer/planner agent.

---

## 1. Scope Recap

- **Read-only codebase understanding tool** — chats about code, commit history, and structure. Never writes code, never opens/reviews PRs.
- **Retrieval strategy:** CRAG (Corrective RAG) for document grading + corrective re-retrieval, blended with Self-RAG-style generation grading (hallucination check + answer-usefulness check).
- **Vector store:** FAISS (local, free, no hosting cost).
- **Orchestration:** LangGraph, with the RAG logic built as its own **subgraph**, invoked as a node from the main graph — matching normal LangGraph subgraph composition, not a monolithic single graph.
- **Checkpointing:** PostgreSQL (`PostgresSaver` / `AsyncPostgresSaver`).
- **Tracing:** LangSmith on every node, both graphs.
- **UI:** Streamlit — chat panel + file tree + commit tree (`git log --graph` style).
- **Models:** free-tier only (Groq, NVIDIA NIM, Hugging Face), detailed in Section 3.

---

## 2. High-Level Graph Composition

Two graphs, composed the normal LangGraph way — the RAG graph is compiled once and used as a **node** inside the main graph (subgraph pattern), not inlined.

```
MAIN GRAPH
──────────
START
  ↓
classify_question   (which source: code / commit / tree-tool / mixed)
  ↓
┌─────────────┬─────────────────┬────────────────┐
▼             ▼                 ▼                ▼
tree_tool   rag_subgraph      rag_subgraph     rag_subgraph
(deterministic  (scope=code)   (scope=commit)   (scope=mixed)
 lookup, no LLM)
  │             │                 │                │
  └─────────────┴─────────────────┴────────────────┘
                        ↓
                  format_response
                        ↓
                       END
```

```
RAG SUBGRAPH (compiled independently, reused for every scope)
───────────────────────────────────────────────────────────
START
  ↓
retrieve                (FAISS similarity search, scoped by source_type filter)
  ↓
grade_documents          (CRAG: per-doc relevance grading)
  ↓
enough relevant docs?
  ├── No  → refine_query (rewrite the query) → retrieve   (corrective loop, max 2 retries)
  └── Yes → generate
                ↓
          grade_generation   (Self-RAG: hallucination check + answers-question check)
                ↓
          passes both checks?
          ├── No, hallucinated → generate again (max 2 retries, then return with a caveat)
          ├── No, doesn't answer → refine_query → retrieve  (loop back)
          └── Yes → END (return grounded answer + citations)
```

This is the same shape as your reference notebook: `TypedDict` state, Pydantic schemas for structured LLM outputs, plain functions as nodes, `add_conditional_edges` for branching, no unnecessary abstraction layers.

---

## 3. Free Model Recommendations

Free-tier availability shifts month to month, so treat this as "what to start with," and always confirm against the live model list before hardcoding an ID (`GET https://api.groq.com/openai/v1/models` for Groq, `build.nvidia.com` catalog page for NVIDIA).

### 3.1 Chat / Generation Models

| Role | Model | Platform | Notes |
|---|---|---|---|
| **Primary reasoning model** (generation, grading, routing) | `openai/gpt-oss-120b` | **Groq** | Free tier, no credit card, fastest inference (LPU hardware). Groq's current flagship open-weight model — also available on NVIDIA NIM as a fallback. |
| **Lightweight/fast tasks** (query rewriting, simple classification) | `openai/gpt-oss-20b` | **Groq** | Smaller, faster, cheaper on rate limits — use for the router and query-refine nodes so your 120B quota is saved for generation/grading. |
| **Fallback #1** (if Groq rate-limited) | `openai/gpt-oss-120b` | **NVIDIA NIM** (`build.nvidia.com`, OpenAI-compatible endpoint `https://integrate.api.nvidia.com/v1`) | Free dev account gives ~1,000 inference credits, 40 req/min limit. Good as an automatic failover, not a primary. |
| **Fallback #2 / alternative reasoning model** | `qwen/qwen3.6-27b` | **Groq** | Groq's other current free-tier flagship; use if `gpt-oss-120b` is unavailable or you want a second opinion for grading nodes. |
| **Avoid** | `llama-3.3-70b-versatile` | Groq | Being phased out on Groq's free/developer tier (deprecated in favor of `gpt-oss-120b`) — don't hardcode this. |

### 3.2 Embedding Model (for FAISS)

Don't spend API quota on embeddings — embed **locally** and keep the LLM calls for reasoning only.

| Role | Model | Platform | Notes |
|---|---|---|---|
| **Primary embeddings** | `BAAI/bge-small-en-v1.5` (or `sentence-transformers/all-MiniLM-L6-v2` if you want smaller/faster) | **Hugging Face**, run locally via `sentence-transformers` — no API key, no rate limit, no cost | Runs fine on CPU for a codebase-sized corpus; this is what actually plugs into FAISS. |
| **Alternative (API-based)** | `nvidia/nv-embedqa-e5-v5` | **NVIDIA NIM** | Only worth it if your laptop struggles with local embedding throughput on a very large repo. |

### 3.3 Practical model-routing note
Wrap model calls behind a small `get_llm(role: str)` factory so a rate-limit failure on Groq can fail over to NVIDIA NIM without touching node logic — this also keeps your LangSmith traces labeled by *role* ("router", "grader", "generator") rather than by provider, which is what actually matters when debugging.

---

## 4. State Schemas

### 4.1 Main Graph State
```python
from typing import TypedDict, List, Literal, Optional, Annotated
import operator
from pydantic import BaseModel, Field

class QuestionClassification(BaseModel):
    scope: Literal["code", "commit", "tree", "mixed"]
    reason: str
    # for "tree" scope — deterministic lookup, no retrieval needed
    tree_query_type: Optional[Literal["file_list", "commit_log", "file_history"]] = None
    tree_target_path: Optional[str] = None

class Citation(BaseModel):
    source_type: Literal["code", "commit", "pr_review"]
    file_path: Optional[str] = None
    commit_sha: Optional[str] = None
    pr_number: Optional[int] = None

class MainState(TypedDict):
    repo_id: str
    question: str
    classification: Optional[QuestionClassification]
    rag_scope: str                 # passed into the RAG subgraph
    tool_result: Optional[str]     # populated by tree_tool node
    answer: str
    citations: List[Citation]
```

### 4.2 RAG Subgraph State (kept separate from MainState — this is what makes it a true reusable subgraph)
```python
class GradedDoc(BaseModel):
    content: str
    source_type: Literal["code", "commit", "pr_review"]
    file_path: Optional[str] = None
    commit_sha: Optional[str] = None
    relevant: bool
    reasoning: str

class GenerationGrade(BaseModel):
    grounded: bool          # Self-RAG hallucination check: is every claim supported by retrieved docs?
    answers_question: bool  # Self-RAG utility check: does it actually answer what was asked?
    reasoning: str

class RagState(TypedDict):
    query: str
    scope: str                      # "code" | "commit" | "mixed" — filters the FAISS search
    retrieved_docs: List[GradedDoc]
    relevant_docs: Annotated[List[GradedDoc], operator.add]
    retry_count: int
    generation: str
    grade: Optional[GenerationGrade]
```

---

## 5. RAG Subgraph — Node-by-Node (CRAG + Self-RAG)

### 5.1 `retrieve`
FAISS similarity search filtered by `scope` (metadata filter on `source_type`). Returns top-k candidates (k=6–8 is a reasonable start) as `GradedDoc` objects with `relevant` unset.

### 5.2 `grade_documents` — CRAG relevance grading
Each retrieved doc is graded independently (this is the CRAG core idea — don't trust the retriever's similarity score alone).

```python
GRADE_DOC_SYSTEM = """You are a relevance grader for a code-and-history retrieval system.

Given a user question and one retrieved document (which may be a code chunk,
a commit message + diff, or a PR review comment), decide if it is actually
relevant to answering the question.

Be strict: a document that merely shares keywords but doesn't help answer
the question is NOT relevant. A document that is genuinely useful context,
even if partial, IS relevant.

Output relevant=true/false and a one-sentence reasoning.
"""
```

### 5.3 Conditional: `enough_relevant_docs`
```python
def enough_relevant_docs(state: RagState) -> str:
    relevant = [d for d in state["relevant_docs"] if d.relevant]
    if len(relevant) >= 2 or state["retry_count"] >= 2:
        return "generate"
    return "refine_query"
```
Capping retries at 2 is a deliberate, defensible scope decision — prevents infinite corrective loops on a genuinely under-indexed question.

### 5.4 `refine_query` — corrective step (CRAG)
```python
REFINE_QUERY_SYSTEM = """You rewrite search queries for a codebase retrieval system.

The previous query did not retrieve enough relevant results. Rewrite it to be
more specific and more likely to match code, commit messages, or file paths
directly — use terminology you'd expect to literally appear in source code or
commit history, not vague natural-language phrasing.

Output only the rewritten query.
"""
```

### 5.5 `generate`
```python
GENERATE_SYSTEM = """You are a senior engineer explaining a codebase to someone
learning it. Answer the question using ONLY the provided context (code chunks,
commit messages/diffs, PR review comments).

Rules:
- Every factual claim must be traceable to a specific provided document.
- Cite sources inline using the format [file_path] or [commit:sha] or [PR #n].
- If the provided context is insufficient to fully answer, say so explicitly —
  do not fill gaps from general programming knowledge presented as fact about
  THIS codebase.
- Be concise and technically precise. No filler.
"""
```

### 5.6 `grade_generation` — Self-RAG-style dual check
```python
GRADE_GENERATION_SYSTEM = """You are checking a generated answer against its source context.

Check two things independently:
1. grounded: Is every specific claim in the answer actually supported by the
   provided context? (hallucination check)
2. answers_question: Does the answer actually address what was asked, not a
   related-but-different question?

Be strict on both.
"""
```

### 5.7 Final conditional
```python
def route_after_grade(state: RagState) -> str:
    grade = state["grade"]
    if not grade.grounded and state["retry_count"] < 2:
        return "generate"          # regenerate from same context
    if not grade.answers_question and state["retry_count"] < 2:
        return "refine_query"      # need different context
    return "END"
```

---

## 6. Main Graph — Node-by-Node

### 6.1 `classify_question`
```python
CLASSIFY_SYSTEM = """You route questions about a GitHub repository to the right
information source.

- "code": questions about what code does, how a function/class works
- "commit": questions about history, why something changed, when it was added,
  what a commit/PR discussed
- "tree": questions about structure — "what files are in X", "show me the
  commit graph", "what's in this directory" — these need a DETERMINISTIC
  lookup, not semantic search
- "mixed": needs both code understanding AND history (e.g. "what does this
  function do and why was it changed")

For "tree" scope, also classify tree_query_type (file_list / commit_log /
file_history) and extract the target path if given.
"""
```

### 6.2 `tree_tool` — deterministic, no LLM
This node never calls a model. It directly queries the pre-built file-tree index and commit-graph index (Section 8) and formats the result as text/structured data for the UI to render. This is the "file and commit tree, like you'd see via `git log --graph`" requirement — handled as data, not retrieval.

### 6.3 `rag_subgraph` node
Simply invokes the compiled RAG subgraph with `scope` set from the classification, and unpacks its output (`generation`, cited docs) back into `MainState`.

### 6.4 `format_response`
Merges `tool_result` (if any) and the RAG subgraph's grounded answer into the final response shown in Streamlit, with citations rendered as clickable references (file path / commit SHA / PR link).

---

## 7. Ingestion: Code, File Tree, and Commit Tree

| Stream | How it's built | Where it's used |
|---|---|---|
| **Code chunks** | Clone repo → walk files → Python `ast`-based function/class chunking (MVP: Python-only) → embed → FAISS index, `source_type="code"` | `retrieve` node in RAG subgraph |
| **Commit history** | `git log --all --pretty=... --numstat` parsed per commit (message, author, date, changed files, diff) → LLM-generated one-line diff summary at ingestion time (embedded alongside raw diff) → FAISS index, `source_type="commit"` | `retrieve` node in RAG subgraph |
| **File tree** | Plain directory walk → stored as a JSON tree (path, type, size) — no embedding | `tree_tool` node, deterministic |
| **Commit tree/graph** | `git log --all --graph --oneline --parents` parsed into a simple DAG structure (commit sha, parent shas, message, author, date) | `tree_tool` node (for text answers) **and** directly rendered in Streamlit (Section 9) |

Ingestion runs as its own pipeline (outside the LangGraph runtime) on repo-add, not per-question — same separation of concerns as your reference notebook's clean node boundaries.

---

## 8. PostgreSQL Checkpointing

Use LangGraph's Postgres checkpointer so conversations and subgraph state survive restarts — this also gives you `thread_id` per chat session, which you've already used in your LangGraph coursework.

```python
from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = "postgresql://user:pass@localhost:5432/codebase_agent"

with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()   # creates the checkpoint tables once
    main_app = main_graph.compile(checkpointer=checkpointer)
```
- Use `AsyncPostgresSaver` instead if your FastAPI layer is async (recommended, since you're already comfortable with FastAPI + async patterns from your MongoDB learning project).
- One `thread_id` per chat session (`chat_sessions.id` from the DB schema in the main plan file) — this is what lets a user resume a conversation about a repo across visits.
- The RAG **subgraph** does not need its own checkpointer — subgraph state is checkpointed as part of the parent graph's state automatically when the parent is compiled with a checkpointer.

---

## 9. Streamlit UI

Three panels, one page:

1. **Chat panel** — standard message history bound to the `thread_id`/checkpoint, streaming the main graph's output.
2. **File tree panel** — rendered from the JSON tree (Section 7) as a collapsible tree widget (`st.expander`-based recursion, or a small custom component). Clicking a file could optionally scope the next question to that file (nice-to-have, not MVP).
3. **Commit tree panel** — this is the "gitlog" view. Two viable approaches, pick one for MVP:
   - **Simplest:** run `git log --all --graph --oneline` and display the raw ASCII output in a `st.code` block — genuinely fine for MVP, zero extra dependencies.
   - **Nicer:** parse the DAG (Section 7) and render it with `graphviz` (Streamlit has native `st.graphviz_chart` support) — one box per commit, edges to parents, hoverable messages. Do this only after the ASCII version works.

Keep the UI itself thin — it's a viewer over graph output and ingestion data, not where any logic lives.

---

## 10. LangSmith Tracing

```python
import os
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "..."
os.environ["LANGSMITH_PROJECT"] = "codebase-intelligence-agent"
```
- Because the RAG graph is a genuine subgraph (compiled separately, invoked as a node), LangSmith will nest its trace under the parent main-graph run automatically — you'll see `classify_question → rag_subgraph{retrieve → grade_documents → generate → grade_generation} → format_response` as one readable trace tree, which is exactly the debugging benefit of building it as a subgraph instead of inlining the nodes.
- Tag runs by `scope` (code/commit/mixed) so you can later filter LangSmith runs to measure, e.g., "how often does the commit-scope path need a corrective retry vs. the code-scope path" — this becomes a real data point for Section 10 of the main plan's evaluation section.

---

## 11. MVP Build Order (this report's version of the earlier plan's Section 13, made concrete)

1. Ingestion: code chunking (Python `ast`) + commit history parsing → FAISS index (local HF embeddings)
2. File tree JSON + commit graph parsing (no LLM involved)
3. RAG subgraph: `retrieve → grade_documents → generate` only — **skip corrective loops and Self-RAG grading in v1**, get the straight-line path working and evaluated first
4. Main graph: `classify_question → rag_subgraph` only — **skip the `tree_tool` branch in v1**, hardcode scope to "mixed"
5. Wire Postgres checkpointer + a single hardcoded repo
6. Streamlit chat panel only (file tree / commit tree panels come after)
7. Build your 10-15 question eval set, measure retrieval precision + groundedness by hand
8. **Only once that's solid:** add the CRAG corrective retry loop, Self-RAG generation grading, the `tree_tool` branch, and the commit-graph Streamlit panel

This mirrors the same discipline as your reference notebook: get the linear happy path fully working before adding conditional branches and fan-out.

---

## 12. Open Decisions Before You Start Coding

- **Model failover:** implement the Groq→NVIDIA fallback from day one, or hardcode Groq only for MVP and add failover in Phase 2? (Recommendation: hardcode Groq only for MVP — failover is real engineering value but not needed to prove the core idea works.)
- **Corrective retry cap:** 2 retries suggested above — tune based on how often your grader actually rejects docs during your eval runs.
- **Commit history depth:** cap ingestion at, e.g., the last 200 commits for MVP — full history on an active repo will slow ingestion and bloat the index without adding much to a "learn this codebase" use case.
- **Where diff summaries are generated:** at ingestion time (recommended — keeps question-answering latency low) vs. on-the-fly at query time (simpler pipeline, slower answers). Ingestion-time is the better default given you're already paying an LLM call per commit either way.
