"""
Checkpoint & Thread / Session Manager
Manages conversational memory persistence (MemorySaver / SqliteSaver) and Thread IDs.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
from langgraph.checkpoint.memory import MemorySaver

from src.config import INDEXES_DIR


import uuid


def generate_thread_id(prefix: str = "session_") -> str:
    """
    Generates a unique UUID-based thread identifier for LangGraph sessions.
    Example: 'session_e7b233a0-4bb8-4fbb-91b4-21946f049079'
    """
    return f"{prefix}{uuid.uuid4()}"


def get_postgres_checkpointer(conn_string: Optional[str] = None):
    """
    Returns a PostgreSQL-backed checkpointer (PostgresSaver) for persistent storage.
    If PostgreSQL is unavailable or not running, gracefully falls back to MemorySaver.
    """
    import os
    import socket
    from urllib.parse import urlparse

    conn_str = conn_string or os.getenv("POSTGRES_URL", "postgresql://postgres:soham@localhost:5442/postgres")
    try:
        # Fast socket probe (1 second max) to prevent blocking if Docker is down
        parsed = urlparse(conn_str)
        host = parsed.hostname or "localhost"
        port = parsed.port or 5442
        with socket.create_connection((host, port), timeout=1.0):
            pass

        from psycopg_pool import ConnectionPool
        from langgraph.checkpoint.postgres import PostgresSaver
        pool = ConnectionPool(conninfo=conn_str, max_size=5, open=True)
        checkpointer = PostgresSaver(pool)
        checkpointer.setup()
        print(f"[CHECKPOINT] Connected to PostgreSQL Checkpointer: {host}:{port}")
        return checkpointer
    except Exception as e:
        print(f"[CHECKPOINT NOTICE] PostgreSQL not reachable ({e}). Falling back to in-memory MemorySaver.")
        return MemorySaver()


def get_checkpointer(use_disk_persistence: bool = False, use_postgres: bool = False):
    """
    Returns an in-memory, SQLite disk, or PostgreSQL checkpointer.
    - MemorySaver (Default): Fast in-memory state tracking using UUID thread IDs.
    - PostgresSaver: Dockerized PostgreSQL persistence on port 5442.
    - SqliteSaver: Local SQLite file persistence.
    """
    if use_postgres:
        return get_postgres_checkpointer()
    if use_disk_persistence:
        try:
            import sqlite3
            from langgraph.checkpoint.sqlite import SqliteSaver
            db_path = INDEXES_DIR / "sessions.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(db_path), check_same_thread=False)
            return SqliteSaver(conn)
        except Exception as e:
            print(f"[CHECKPOINT WARNING] Could not initialize SqliteSaver ({e}), falling back to MemorySaver.")
            return MemorySaver()
    return MemorySaver()


def create_thread_config(thread_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Generates the standard LangGraph RunnableConfig dict for a thread/session using UUID.
    """
    if not thread_id:
        thread_id = generate_thread_id()
    return {
        "configurable": {
            "thread_id": thread_id
        }
    }


def retrieve_all_threads(checkpointer) -> List[str]:
    """
    Retrieves all unique thread IDs stored in the checkpointer.
    """
    all_threads = set()
    try:
        for checkpoint in checkpointer.list(None):
            cfg = getattr(checkpoint, "config", {})
            if isinstance(cfg, dict):
                tid = cfg.get("configurable", {}).get("thread_id")
                if tid:
                    all_threads.add(str(tid))
    except Exception as e:
        print(f"[THREAD RETRIEVAL WARNING] {e}")
    return list(all_threads)


def get_thread_history(app, thread_id: str) -> list:
    """
    Retrieves the chronological state history for a given thread_id.
    """
    config = create_thread_config(thread_id)
    try:
        return list(app.get_state_history(config))
    except Exception as e:
        print(f"[HISTORY ERROR] Could not get state history: {e}")
        return []
