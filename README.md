# 🔍 RepoLens

**RepoLens** is an intelligent codebase exploration, semantic search, and AI-driven codebase assistant designed to analyze, index, and query repositories with context-aware RAG pipelines and multi-agent workflows.

---

## 🚀 Features (In Development)

- 🧠 **Codebase Ingestion & AST Parsing**: Index syntax trees, symbols, classes, functions, and docstrings.
- ⚡ **Hybrid RAG & Retrieval**: Vector search combined with BM25 / symbol-aware code retrieval.
- 🤖 **Agentic Code Exploration**: Multi-step query planning, code navigation, and explanation agents.
- 📊 **Architecture & Dependency Mapping**: Automated dependency graph creation and high-level architectural insights.

---

## 🛠️ Getting Started

### Prerequisites

- Python 3.10+
- Virtual environment tool (`venv` or `uv`)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/sohamsabhaya/RepoLens.git
   cd RepoLens
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables:
   ```bash
   cp .env.example .env
   # Update your API keys in .env
   ```

---

## 📁 Project Structure

```
RepoLens/
├── src/
│   ├── agent/         # Multi-agent workflows & reasoning
│   ├── ingestion/     # Code parsing, AST extraction & chunking
│   ├── rag/           # Embeddings, retrieval & vector store
│   └── config.py      # App configurations & settings
├── notebooks/         # Prototyping and experiments
├── tests/             # Unit and integration test suites
├── main.py            # Main entrypoint
├── requirements.txt   # Dependencies
├── .env.example       # Template environment variables
└── README.md
```

---

## 📄 License

MIT License
