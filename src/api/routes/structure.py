"""
Deterministic Structure & Git DAG Routes
"""

from fastapi import APIRouter, HTTPException

from src.ingestion.tree_indexer import build_file_tree, build_commit_dag, format_file_tree_as_text, format_commit_dag_as_text
from src.api.schemas import TreeResponse, CommitDagResponse

router = APIRouter(prefix="/api", tags=["Codebase Structure & Git DAG"])


@router.get("/tree", response_model=TreeResponse)
def get_file_tree():
    """
    Returns deterministic file tree without LLM hallucinations.
    """
    from src.api.main import app_state

    if not app_state.active_repo_path or not app_state.active_repo_path.exists():
        raise HTTPException(status_code=400, detail="No active repository path found.")
    
    tree_data = build_file_tree(app_state.active_repo_path)
    formatted_tree = format_file_tree_as_text(app_state.active_repo_path)
    return TreeResponse(
        repo_path=str(app_state.active_repo_path),
        tree_json=tree_data,
        formatted_tree=formatted_tree,
    )


@router.get("/commits", response_model=CommitDagResponse)
def get_commit_dag(limit: int = 50):
    """
    Returns deterministic Git commit DAG history.
    """
    from src.api.main import app_state

    if not app_state.active_repo_path or not app_state.active_repo_path.exists():
        raise HTTPException(status_code=400, detail="No active repository path found.")
    
    dag_data = build_commit_dag(app_state.active_repo_path, max_commits=limit)
    formatted_dag = format_commit_dag_as_text(app_state.active_repo_path, max_commits=limit)
    return CommitDagResponse(
        repo_path=str(app_state.active_repo_path),
        commit_count=len(dag_data),
        commits=dag_data,
        formatted_dag=formatted_dag,
    )
