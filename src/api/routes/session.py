"""
Session Persistence & Time-Travel Routes
"""

from fastapi import APIRouter, HTTPException

from src.agent.checkpoint import retrieve_all_threads, get_thread_history

router = APIRouter(prefix="/api", tags=["Sessions & Time-Travel"])


@router.get("/threads")
def list_session_threads():
    """
    Returns all active UUID thread IDs.
    """
    from src.api.main import app_state
    threads = retrieve_all_threads(app_state.checkpointer)
    return {"threads": threads}


@router.get("/history/{thread_id}")
def get_session_history(thread_id: str):
    """
    Returns chronological state checkpoints for time-travel inspection.
    """
    from src.api.main import app_state

    if not app_state.agent_app:
        raise HTTPException(status_code=400, detail="Agent graph not initialized.")
    
    history = get_thread_history(app_state.agent_app, thread_id)
    serialized_history = []
    for snapshot in history:
        serialized_history.append({
            "checkpoint_id": snapshot.config.get("configurable", {}).get("checkpoint_id", "unknown"),
            "values": {
                "intent": snapshot.values.get("intent"),
                "messages_count": len(snapshot.values.get("messages", [])),
                "last_message": snapshot.values.get("messages", [])[-1].content if snapshot.values.get("messages") else None,
            },
            "next": snapshot.next,
        })
    return {"thread_id": thread_id, "checkpoints": serialized_history}
