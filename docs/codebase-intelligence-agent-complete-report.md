# Codebase Intelligence Agent — Complete Technical Report

**A read-only, citable RAG system for understanding a GitHub repository's code, commit history, and structure — built with LangGraph subgraphs, LangChain CRAG/Self-RAG retrieval, FAISS, free-tier models, and PostgreSQL checkpointing.**

This report merges the original plan and the first technical report into one complete, buildable reference. It keeps the same philosophy as your reference notebook — a simple, linear flow first, expanded only where the design genuinely needs branching — but documents every layer end to end: graphs, state, prompts, ingestion, database, models, and UI.

---

## Part I — Scope and Design Philosophy

### 1.1 What this system is

A chat interface over a single ingested repository that answers three kinds of questions:

- **What does this code do** — grounded in the actual source.
- **Why does it exist / when did it change** — grounded in commit history and diffs.
- **How is it structured** — file tree and commit tree, answered deterministically, not guessed by an LLM.

Every answer is grounded and cited (file path, commit SHA, or PR number). If the retrieved context doesn't support an answer, the system says so rather than filling gaps from general knowledge.

### 1.2 What this system deliberately is not

It never writes code, never opens a PR, never reviews a PR against a style guide, and never executes anything against the repository. It is a **read-only explainer**, which is what keeps it structurally distinct from "fetch issue → solve → PR" agents and keeps its evaluation story honest (a wrong explanation is embarrassing; it can't break a build).

### 1.3 Core design principles carried through every section below

1. **Two graphs, not one.** A main orchestration graph and a RAG subgraph, compiled independently and composed the normal LangGraph way (the RAG graph is a *node* in the main graph). This is what makes CRAG/Self-RAG reusable across every retrieval scope instead of being copy-pasted logic.
2. **Structure isn't retrieval.** File trees and commit graphs are deterministic lookups over a JSON/DAG structure — never embedded, never something an LLM "guesses" at.
3. **Every source gets source-appropriate handling.** Code is chunked at function/class boundaries; commits are one record each; diffs get an LLM-generated natural-language summary at ingestion time so they become searchable by intent ("when was auth added") rather than only by literal diff text.
4. **Free-tier only, but swappable.** Every model call goes through a small factory keyed by *role* (router, grader, generator, embedder), not hardcoded by provider, so a rate limit or a provider outage never touches node logic.
5. **Everything traced.** LangSmith wraps both graphs; because the RAG graph is a true subgraph, traces nest automatically into a single readable tree.
6. **Start linear, add branches only once evaluated.** The MVP skips the query router, the corrective retry loop, and the tree-tool branch. Each is added only after the straight-line path has real precision/faithfulness numbers behind it.

---

## Part II — System Architecture

### 2.1 End-to-end system diagram

```
GitHub Repo URL
      │
      ▼
 Ingestion Pipeline (runs once per repo-add, outside the LangGraph runtime)
  ├── Code ingestor        (clone + ast/tree-sitter chunking)
  ├── Commit ingestor      (git log --numstat + diff parsing + LLM diff summaries)
  ├── PR/review ingestor   (GitHub REST API — Phase 2)
  ├── File-tree indexer    (directory walk → JSON tree, no embedding)
  └── Commit-tree indexer  (git log --all --graph --parents → DAG, no embedding)
      │
      ▼
 Chunking + Metadata (source_type, file path, commit SHA, PR #, author, date)
      │
      ▼
 Embedding + FAISS Indexing (separate index per source_type: code / commit / pr_review)
      │
      ▼
 FastAPI backend (ingest endpoint, ask endpoint, tree/graph endpoints)
      │
      ▼
 LangGraph Orchestrator
   MAIN GRAPH → classify_question → {tree_tool | rag_subgraph} → format_response
   RAG SUBGRAPH → retrieve → grade_documents → (corrective loop) → generate → grade_generation
      │
      ▼
 PostgreSQL — repo metadata, chat sessions, messages, LangGraph checkpoints
      │
      ▼
 Streamlit UI — chat panel, file-tree panel, commit-tree panel (gitlog-style)
      │
      ▼
 LangSmith — every node in both graphs traced, nested automatically
```

### 2.2 Main graph — question routing and composition

```
START
  ↓
classify_question   (LLM: which source — code / commit / tree / mixed)
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

The RAG subgraph is **compiled once** and reused across every scope by passing `scope` into its input state — it is not re-defined per branch. This is the actual payoff of building it as a subgraph rather than inlining three near-identical node chains.

### 2.3 RAG subgraph — CRAG (corrective retrieval) + Self-RAG (generation grading)

```
START
  ↓
retrieve                 (FAISS similarity search, metadata-filtered by scope)
  ↓
grade_documents           (CRAG: per-document relevance grading, independent of similarity score)
  ↓
enough relevant docs?
  ├── No  → refine_query (LLM rewrites the query) → retrieve   (max 2 corrective retries)
  └── Yes → generate                                            (grounded answer + citations)
                ↓
          grade_generation   (Self-RAG: hallucination check + answers-question check)
                ↓
          passes both checks?
          ├── grounded=false        → generate again (max 2 retries, then return with a caveat)
          ├── answers_question=false → refine_query → retrieve  (different context needed)
          └── both pass             → END
```

**Why CRAG here specifically:** a retriever's cosine-similarity score is not a relevance guarantee, especially across three structurally different document types (prose-like commit summaries, terse code, threaded review comments). CRAG's independent relevance grade is what catches "high similarity, wrong content" before it reaches generation.

**Why Self-RAG on top of CRAG:** CRAG grades *inputs*; Self-RAG grades the *output*. A generation can still hallucinate or drift off-topic even from good context — the dual grounded/answers_question check catches both failure modes separately, which matters because the fix for each is different (regenerate vs. re-retrieve).

---

## Part III — LangChain / LangGraph Implementation Details

### 3.1 Why LangChain components, specifically

- `FAISS` vector store class (`langchain_community.vectorstores.FAISS`) — local, free, metadata-filterable via `similarity_search(query, k, filter={...})`.
- `PydanticOutputParser` / native structured output (`with_structured_output`) for every grading and classification node — CRAG and Self-RAG both live or die on getting clean, typed grades back from the LLM, not free-text you have to regex.
- `RecursiveCharacterTextSplitter` is deliberately **not** used for code — see Section 5 on chunking. It is fine for PR review comment bodies, which are closer to prose.
- `PostgresSaver` / `AsyncPostgresSaver` from `langgraph.checkpoint.postgres` for durable, resumable conversations.
- `langsmith` tracing enabled globally via environment variables — no manual span code needed if `LANGSMITH_TRACING=true` is set before graph compilation.

### 3.2 State schemas

**Main graph state**
```python
from typing import TypedDict, List, Literal, Optional
from pydantic import BaseModel, Field

class QuestionClassification(BaseModel):
    scope: Literal["code", "commit", "tree", "mixed"]
    reason: str
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
    rag_scope: str
    tool_result: Optional[str]
    answer: str
    citations: List[Citation]
```

**RAG subgraph state** — kept intentionally separate from `MainState`. This separation is what makes the subgraph genuinely reusable rather than a function that happens to share a state object with the parent.
```python
import operator
from typing import Annotated

class GradedDoc(BaseModel):
    content: str
    source_type: Literal["code", "commit", "pr_review"]
    file_path: Optional[str] = None
    commit_sha: Optional[str] = None
    relevant: bool
    reasoning: str

class GenerationGrade(BaseModel):
    grounded: bool
    answers_question: bool
    reasoning: str

class RagState(TypedDict):
    query: str
    scope: str
    retrieved_docs: List[GradedDoc]
    relevant_docs: Annotated[List[GradedDoc], operator.add]
    retry_count: int
    generation: str
    grade: Optional[GenerationGrade]
```

### 3.3 Graph composition (subgraph pattern)

```python
from langgraph.graph import StateGraph, START, END

# --- Build and compile the RAG subgraph independently ---
rag_builder = StateGraph(RagState)
rag_builder.add_node("retrieve", retrieve_node)
rag_builder.add_node("grade_documents", grade_documents_node)
rag_builder.add_node("refine_query", refine_query_node)
rag_builder.add_node("generate", generate_node)
rag_builder.add_node("grade_generation", grade_generation_node)

rag_builder.add_edge(START, "retrieve")
rag_builder.add_edge("retrieve", "grade_documents")
rag_builder.add_conditional_edges("grade_documents", enough_relevant_docs,
                                   {"generate": "generate", "refine_query": "refine_query"})
rag_builder.add_edge("refine_query", "retrieve")
rag_builder.add_edge("generate", "grade_generation")
rag_builder.add_conditional_edges("grade_generation", route_after_grade,
                                   {"generate": "generate", "refine_query": "refine_query", "END": END})

rag_graph = rag_builder.compile()   # compiled once, reused for every scope

# --- Main graph invokes the compiled RAG graph as a plain node ---
def rag_subgraph_node(state: MainState) -> dict:
    result = rag_graph.invoke({
        "query": state["question"],
        "scope": state["rag_scope"],
        "retrieved_docs": [], "relevant_docs": [], "retry_count": 0,
        "generation": "", "grade": None,
    })
    return {"answer": result["generation"], "citations": extract_citations(result["relevant_docs"])}

main_builder = StateGraph(MainState)
main_builder.add_node("classify_question", classify_question_node)
main_builder.add_node("tree_tool", tree_tool_node)
main_builder.add_node("rag_subgraph", rag_subgraph_node)
main_builder.add_node("format_response", format_response_node)

main_builder.add_edge(START, "classify_question")
main_builder.add_conditional_edges("classify_question", route_by_scope,
                                    {"tree": "tree_tool", "code": "rag_subgraph",
                                     "commit": "rag_subgraph", "mixed": "rag_subgraph"})
main_builder.add_edge("tree_tool", "format_response")
main_builder.add_edge("rag_subgraph", "format_response")
main_builder.add_edge("format_response", END)
```

This is the same shape as your reference notebook: `TypedDict` + Pydantic state, plain functions as nodes, `add_conditional_edges` for branching, no unnecessary abstraction layers — just applied to a retrieval-and-explain agent instead of a writer/planner agent.

---

## Part IV — Prompts (High-Level, Per Node)

Keep every system prompt as a named constant, short, and role-scoped — one job per prompt, not a kitchen-sink instruction block.

### 4.1 `classify_question`
```
You route questions about a GitHub repository to the right information source.

- "code": what code does, how a function/class works
- "commit": history, why something changed, when it was added, what a
  commit/PR discussed
- "tree": structure questions — "what files are in X", "show the commit
  graph", "what's in this directory" — these need a DETERMINISTIC lookup,
  not semantic search
- "mixed": needs both code understanding AND history

For "tree" scope, also classify tree_query_type (file_list / commit_log /
file_history) and extract the target path if one is given.
```

### 4.2 `grade_documents` (CRAG relevance grading)
```
You are a relevance grader for a code-and-history retrieval system.

Given a user question and one retrieved document (a code chunk, a commit
message + diff, or a PR review comment), decide if it is genuinely relevant
to answering the question.

Be strict: a document that only shares keywords but doesn't help answer the
question is NOT relevant. A document that is useful context, even partially,
IS relevant.

Output relevant=true/false and one sentence of reasoning.
```

### 4.3 `refine_query` (CRAG corrective rewrite)
```
The previous query did not retrieve enough relevant results. Rewrite it to
be more specific and more likely to match code, commit messages, or file
paths directly — use terms you'd expect to literally appear in source code
or commit history, not vague natural-language phrasing.

Output only the rewritten query.
```

### 4.4 `generate`
```
You are a senior engineer explaining a codebase to someone learning it.
Answer using ONLY the provided context (code chunks, commit messages/diffs,
PR review comments).

Rules:
- Every factual claim must trace to a specific provided document.
- Cite inline as [file_path], [commit:sha], or [PR #n].
- If context is insufficient to fully answer, say so explicitly — never
  fill gaps from general programming knowledge presented as fact about
  THIS codebase.
- Be concise and technically precise. No filler.
```

### 4.5 `grade_generation` (Self-RAG dual check)
```
Check two things independently, against the generated answer and its
source context:

1. grounded — is every specific claim in the answer actually supported by
   the provided context? (hallucination check)
2. answers_question — does the answer address what was actually asked, not
   a related-but-different question?

Be strict on both. Output both booleans plus one sentence of reasoning.
```

### 4.6 Diff summarization (ingestion-time, not a graph node)
```
Summarize what this diff does in one or two plain-language sentences,
focused on intent (what changed and why it likely matters), not a
line-by-line description. This summary will be embedded so someone can
find this commit by asking a natural-language question about the codebase.
```

### 4.7 Conditional-edge functions
```python
def enough_relevant_docs(state: RagState) -> str:
    relevant = [d for d in state["relevant_docs"] if d.relevant]
    if len(relevant) >= 2 or state["retry_count"] >= 2:
        return "generate"
    return "refine_query"

def route_after_grade(state: RagState) -> str:
    grade = state["grade"]
    if not grade.grounded and state["retry_count"] < 2:
        return "generate"
    if not grade.answers_question and state["retry_count"] < 2:
        return "refine_query"
    return "END"
```
Capping retries at 2 is a deliberate scope decision, not an oversight — it prevents infinite corrective loops on a genuinely under-indexed question, and gives you a clean number to report in the evaluation section ("X% of questions needed a corrective retry").

---

## Part V — Ingestion: Code, File Tree, and Commit Tree

| Stream | How it's built | Storage | Used by |
|---|---|---|---|
| **Code chunks** | Clone → walk files → `ast` (Python MVP) / `tree-sitter` (later, multi-language) chunking at function/class boundaries. Each chunk keeps `file_path`, `function/class name`, `language`. | FAISS index, `source_type="code"` | `retrieve` node, filtered by scope |
| **Commit history** | `git log --all --pretty=<fmt> --numstat`, parsed per commit into one record: message, author, date, changed files, diff. An LLM generates a one/two-sentence diff summary at ingestion time (Section 4.6), embedded *alongside* the raw diff — this is what makes commits searchable by intent ("when was auth added") instead of only by literal diff text. | FAISS index, `source_type="commit"` | `retrieve` node |
| **PR reviews** *(Phase 2)* | GitHub REST API — PR description + review comments, each kept with thread context (what code it comments on, resolved/unresolved). | FAISS index, `source_type="pr_review"` | `retrieve` node |
| **File tree** | Plain directory walk → JSON tree (path, type, size). **Not embedded.** | Postgres/JSON blob | `tree_tool`, deterministic |
| **Commit tree/graph** | `git log --all --graph --oneline --parents`, parsed into a simple DAG (commit SHA, parent SHAs, message, author, date). **Not embedded.** | Postgres/JSON blob | `tree_tool` (text answers) **and** rendered directly in Streamlit |

**Why naive fixed-length chunking fails here:** splitting code like a PDF breaks functions mid-body and destroys meaning; a diff is structurally add/remove-lines-against-a-path, not prose, so it needs a generated summary to become semantically searchable at all. Treating code, commits, diffs, and file/commit structure as four differently-shaped problems — rather than one blob to embed — is the actual technical difficulty this project is solving, and the reason it's a stronger portfolio piece than a single-vector-store code chatbot.

Ingestion runs once per repo-add, **outside** the LangGraph runtime — same separation of concerns as the rest of the design: graphs answer questions, ingestion prepares the data they answer from.

---

## Part VI — Free Model Stack

Free-tier availability shifts monthly. Treat the table below as "what to start building against today," and always confirm the live catalog before hardcoding a model ID — `GET https://api.groq.com/openai/v1/models` for Groq, and the `build.nvidia.com` model catalog page for NVIDIA NIM. One structural note worth knowing while you plan model failover: **NVIDIA acquired Groq in a deal that closed in December 2025**, with most of Groq's engineering team moving to NVIDIA; GroqCloud itself continues operating independently on its own LPU hardware for now, so Groq and NVIDIA NIM remain two separate free endpoints in practice, but don't be surprised if they consolidate further — check both catalogs before you lock in an architecture that assumes they'll always be independent.

### 6.1 Chat / generation / grading / routing models

| Role | Model | Platform | Notes |
|---|---|---|---|
| **Primary reasoning model** — generation, grading, routing | `openai/gpt-oss-120b` | **Groq** (`api.groq.com/openai/v1`) | Free tier, no credit card, LPU-speed inference. Also present on NVIDIA NIM as a fallback path. |
| **Lightweight/fast tasks** — query classification, query rewriting | `openai/gpt-oss-20b` or `llama-3.1-8b-instant` | **Groq** | Smaller and by far the most permissive free-tier model on Groq (highest daily request cap) — route the router and `refine_query` nodes here to save your 120B quota for generation/grading. |
| **Fallback #1** (Groq rate-limited) | `openai/gpt-oss-120b` | **NVIDIA NIM** (`https://integrate.api.nvidia.com/v1`) | Free developer account, ~1,000 inference credits, roughly 40 req/min ceiling (best-effort, shared infrastructure, not an SLA). Good as automatic failover, not primary. |
| **Fallback #2 / coding-specialized alternative** | `qwen/qwen3-coder-480b` | **NVIDIA NIM** | A large, agentic-coding-tuned model — genuinely useful as the "fine-tuned for codebase analysis" model if you want the `generate` node to reason more sharply about code specifically, rather than sharing one general-purpose model across every scope. |
| **Second general-purpose fallback** | `qwen3.6-27b` (or current Qwen flagship on Groq) | **Groq** | Use if `gpt-oss-120b` is unavailable, or as a second opinion specifically for grading nodes (CRAG/Self-RAG benefit from a model that isn't the same one that generated the content it's grading). |
| **Avoid** | `llama-3.3-70b-versatile` | Groq | Being phased out on Groq's free tier in favor of `gpt-oss-120b` — don't hardcode this. |

### 6.2 Embedding models (feed FAISS)

Don't spend API quota on embeddings if you don't have to — embed **locally** and reserve API calls for reasoning.

| Role | Model | Platform | Notes |
|---|---|---|---|
| **Primary embeddings (general + code)** | `BAAI/bge-small-en-v1.5` or `sentence-transformers/all-MiniLM-L6-v2` | **Hugging Face**, run locally via `sentence-transformers` | No API key, no rate limit, no cost. Runs fine on CPU at codebase scale. This is what actually plugs into FAISS for both code chunks and commit records. |
| **Code-specialized embeddings (optional upgrade)** | `nvidia/nv-embedcode-7b-v1` | **NVIDIA NIM** | Purpose-built for code retrieval — a genuine "fine-tuned for embeddings" option if you want a second, code-specific FAISS index instead of reusing one general-purpose embedder across code and prose (commits/reviews). Worth it once you're past MVP and want to squeeze retrieval precision. |
| **General-purpose API embeddings (fallback)** | `nvidia/nv-embedqa-e5-v5` or `nvidia/nemotron-3-embed-1b` | **NVIDIA NIM** (free endpoint) | Only worth it if local embedding throughput becomes a bottleneck on a very large repo, or you want to avoid running `sentence-transformers` locally at all. |

### 6.3 Model-selection summary against your ask

You asked for one model per role (embeddings / codebase analysis / text generation) where a fine-tuned option exists, with the fallback that one model everywhere is fine. Mapped concretely:

- **Embeddings:** `BAAI/bge-small-en-v1.5` locally (default) → `nvidia/nv-embedcode-7b-v1` if you want a code-specialized second index later.
- **Codebase analysis** (the `generate` node when scope is code-heavy): `qwen/qwen3-coder-480b` on NVIDIA NIM — genuinely coding-tuned, not just general-purpose.
- **General text generation / routing / grading:** `openai/gpt-oss-120b` on Groq, with `gpt-oss-20b` for the cheap/fast nodes.
- **If you'd rather not manage three model identities:** `openai/gpt-oss-120b` alone, on Groq, handles routing, grading, and generation acceptably for MVP — this is a legitimate simplification, not a compromise that breaks the design.

### 6.4 Model-routing implementation pattern

```python
def get_llm(role: str):
    """
    role ∈ {"router", "grader", "generator_general", "generator_code"}
    Wrap provider selection here so a Groq rate-limit failure fails over
    to NVIDIA NIM without touching any node's logic, and so LangSmith
    traces are labeled by role rather than by provider.
    """
    ...
```
Tag every LLM call in LangSmith by `role`, not by provider — that's the label that actually matters when you're debugging *why* a node made a bad decision, versus which company happened to serve the tokens.

---

## Part VII — PostgreSQL

### 7.1 Schema

- `repositories` — id, url, name, ingested_at, default_branch
- `ingestion_jobs` — id, repo_id, status, source_type, started_at, completed_at
- `chat_sessions` — id, repo_id, created_at
- `messages` — id, session_id, role, content, cited_sources (JSON), created_at
- LangGraph checkpoint tables — created automatically by `PostgresSaver.setup()`

### 7.2 Checkpointer wiring

```python
from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = "postgresql://user:pass@localhost:5432/codebase_agent"

with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()   # creates checkpoint tables once
    main_app = main_graph.compile(checkpointer=checkpointer)
```

- Use `AsyncPostgresSaver` if the FastAPI layer is async — the recommended path given the rest of the stack is async-friendly.
- One `thread_id` per `chat_sessions.id` — this is what lets a user resume a conversation about a repo across visits.
- The RAG **subgraph does not need its own checkpointer** — its state is checkpointed automatically as part of the parent graph's state whenever the parent graph is compiled with a checkpointer. This is one of the concrete practical benefits of the subgraph pattern, not just an organizational nicety.

---

## Part VIII — Streamlit UI

Three panels on one page, kept thin — the UI is a viewer over graph output and ingestion data, not where any logic lives.

1. **Chat panel** — standard message history bound to `thread_id`, streaming the main graph's output token-by-token if the client supports it.
2. **File-tree panel** — rendered from the JSON tree (Part V) as a collapsible tree (`st.expander`-based recursion, or a small custom component). Clicking a file can optionally scope the next question to that file — nice-to-have, not MVP.
3. **Commit-tree panel** — the "gitlog" view you asked for, with two viable approaches:
   - **Simplest (MVP):** run `git log --all --graph --oneline` and display the raw ASCII output in an `st.code` block. Zero extra dependencies, genuinely fine to ship first.
   - **Nicer (post-MVP):** parse the DAG built in Part V and render it with `graphviz` (`st.graphviz_chart` is native to Streamlit) — one box per commit, edges to parents, hoverable commit messages.

Both the file tree and commit tree are answered by the **same deterministic `tree_tool` node** when asked about in chat ("what files are in `/src/auth`", "show me the commit graph for this module") — the Streamlit panels are a second, always-visible presentation of the same underlying JSON/DAG structures, not a separate data path.

---

## Part IX — LangSmith Tracing

```python
import os
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "..."
os.environ["LANGSMITH_PROJECT"] = "codebase-intelligence-agent"
```

- Because the RAG graph is a genuine subgraph — compiled separately, invoked as a node — LangSmith nests its trace under the parent run automatically. You get one readable tree: `classify_question → rag_subgraph{retrieve → grade_documents → generate → grade_generation} → format_response`, instead of a flat list of spans you have to mentally reassemble.
- Tag every run by `scope` (code / commit / mixed) so you can later filter in LangSmith to measure things like "how often does the commit-scope path need a corrective retry versus the code-scope path" — that becomes a real, citable data point in the evaluation section below.
- Tag every LLM call by `role` (Part VI.4) so provider swaps and failovers don't fragment your traces by vendor.

---

## Part X — Evaluation

This is the section that separates a defensible portfolio project from a demo GIF.

Build a small hand-labeled test set on 1–2 real repos you know well.

- **Retrieval accuracy:** for 15–20 questions, label which source(s) *should* be retrieved from; measure router accuracy (did `classify_question` pick correctly?) and retrieval precision/recall within the chosen source.
- **Faithfulness:** does the generated answer's claim actually match the cited code/commit/PR comment, or is it a plausible-sounding hallucination? Manual check against citations, optionally cross-checked with LLM-as-judge on a subset you verify by hand.
- **Multi-hop correctness:** for "mixed" questions (e.g. "what does X do and why was it changed"), does the system actually pull from both code and commit sources, or silently answer from only one?
- **Corrective-loop and grading stats:** what fraction of questions triggered a CRAG retry? What fraction triggered a Self-RAG regeneration versus a re-retrieve? These numbers come straight out of LangSmith tags (Part IX) and are genuinely informative, not decorative.
- **Honest reporting:** include the failure cases — where the router picked the wrong source, where a diff summary missed context, where two corrective retries still weren't enough. This is what makes the metrics credible.

---

## Part XI — MVP Build Order

Build the linear path first. Skip every branch until the straight line is evaluated and solid.

1. Ingestion: code chunking (`ast`, Python-only) + commit history parsing (message + diff, no PR reviews yet) → single FAISS index over both, using local Hugging Face embeddings.
2. File tree JSON + commit graph parsing — no LLM involved, built alongside ingestion.
3. RAG subgraph, minimal: `retrieve → generate` only. **Skip `grade_documents`, the corrective loop, and `grade_generation` in v1** — get the straight-line retrieve-and-answer path working and evaluated first.
4. Main graph, minimal: `classify_question → rag_subgraph` only. **Skip the `tree_tool` branch in v1** — hardcode scope to `"mixed"` so every question hits the same single combined index.
5. Wire the Postgres checkpointer against one hardcoded repo.
6. Streamlit chat panel only — file-tree and commit-tree panels come after.
7. Build the 10–15 question eval set on that same repo; measure retrieval precision and groundedness by hand.
8. **Only once that's solid:** add CRAG's `grade_documents` + corrective retry loop, Self-RAG's `grade_generation`, the `tree_tool` branch and query router, PR/review ingestion, and the commit-graph Streamlit panel.

### Phased roadmap (post-MVP)

| Phase | Adds |
|---|---|
| MVP | Code + commit ingestion, single FAISS index, linear retrieve→generate, eval set |
| 2 | Query router (`classify_question`), separate FAISS indices per source type, PR/review ingestion via GitHub API |
| 3 | `tree_tool` deterministic branch, diff summarization at ingestion, CRAG corrective loop, Self-RAG generation grading |
| 4 | Hybrid retrieval (vector + BM25), reranking, multi-language code parsing via `tree-sitter` |
| 5 | Multi-repo support, auth, Docker, CI/CD, cloud deployment |

---

## Part XII — Open Decisions Before You Start Coding

- **Model failover:** wire the Groq → NVIDIA fallback from day one, or hardcode Groq-only for MVP and add failover in Phase 2? *Recommendation:* Groq-only for MVP — failover is real engineering value but isn't needed to prove the core retrieval idea works.
- **One model vs. role-specialized models:** a single `gpt-oss-120b` for every role is a legitimate MVP simplification (Part VI.3); split into a code-specialized generator (`qwen3-coder-480b`) and a code-specialized embedder (`nv-embedcode-7b-v1`) only once you have eval numbers showing the general-purpose setup is the bottleneck.
- **Corrective retry cap:** 2 retries is a reasonable starting default — tune it against how often your grader actually rejects documents during real eval runs, and report the tuned number as a finding, not an assumption.
- **Commit history depth:** cap ingestion at, e.g., the last 200 commits for MVP. Full history on an active repo slows ingestion and adds noise without adding much value to a "learn this codebase" use case — a capped scope is defensible, not a shortcut you need to hide.
- **Diff summaries — ingestion-time vs. query-time:** ingestion-time (recommended) keeps question-answering latency low, since you're already paying one LLM call per commit either way during ingestion.
- **Single hardcoded repo vs. accept-any-URL from day one:** hardcoding one repo is faster to build and evaluate; accepting arbitrary URLs is a more impressive demo but adds real edge cases (private repos, huge monorepos, non-Python languages) — decide based on whether the eval story or the demo story matters more for how this project will be shown.
- **Chroma-style local FAISS index vs. a hosted vector DB:** FAISS local is the default throughout this report (free, no setup cost); revisit only if you need a deployable multi-user demo rather than a local working build.
