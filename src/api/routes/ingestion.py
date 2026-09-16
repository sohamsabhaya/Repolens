"""
Repository Ingestion Route
"""

from pathlib import Path
from fastapi import APIRouter, HTTPException

from src.config import CLONED_REPOS_DIR
from src.agent.models import get_reasoning_llm, get_embeddings_model
from src.agent.checkpoint import get_checkpointer
from src.agent.main_graph import create_code_agent
from src.rag.vectorstore import load_or_build_vectorstore
from src.api.schemas import IngestRequest, IngestResponse

router = APIRouter(prefix="/api", tags=["Ingestion"])


@router.post("/ingest", response_model=IngestResponse)
def ingest_repository(req: IngestRequest):
    """
    Ingests, parses AST code chunks, Git history commits/diffs, and builds local FAISS index.
    """
    from src.api.main import app_state
    try:
        embeddings = get_embeddings_model()
        vstore = load_or_build_vectorstore(
            repo_source=req.repo_source,
            embeddings=embeddings,
            force_reindex=req.force_reindex,
        )
        app_state.vectorstore = vstore
        app_state.active_repo_source = req.repo_source

        if req.repo_source.startswith(("http://", "https://", "git@")):
            repo_name = req.repo_source.rstrip("/").split("/")[-1].replace(".git", "")
            app_state.active_repo_path = CLONED_REPOS_DIR / repo_name
        else:
            app_state.active_repo_path = Path(req.repo_source).resolve()

        llm = get_reasoning_llm()
        app_state.agent_app = create_code_agent(
            llm=llm,
            vectorstore=vstore,
            checkpointer=app_state.checkpointer,
            repo_path=app_state.active_repo_path,
        )

        doc_count = vstore.index.ntotal if hasattr(vstore, "index") else 0

        return IngestResponse(
            status="success",
            repo_source=req.repo_source,
            resolved_path=str(app_state.active_repo_path),
            total_docs_indexed=doc_count,
            message="Repository ingested and vectorstore initialized successfully.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
