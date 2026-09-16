"""
RepoLens — FastAPI Main Application
Modular REST & Streaming Server for Agentic Codebase Intelligence
"""

from pathlib import Path
from typing import Optional, Any
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.agent.checkpoint import get_postgres_checkpointer, get_checkpointer
from src.api.routes import ingestion, chat, structure, session, explorer


class AppState:
    vectorstore: Optional[Any] = None
    agent_app: Optional[Any] = None
    checkpointer: Any = None
    active_repo_path: Optional[Path] = None
    active_repo_source: Optional[str] = None


app_state = AppState()
# Uses PostgreSQL if available, otherwise MemorySaver
app_state.checkpointer = get_postgres_checkpointer()

app = FastAPI(
    title="RepoLens API",
    description="Enterprise Multi-Turn Codebase Intelligence & Git History Exploration API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
STATIC_DIR = Path.cwd() / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Register modular routers
app.include_router(ingestion.router)
app.include_router(chat.router)
app.include_router(structure.router)
app.include_router(session.router)
app.include_router(explorer.router)


@app.get("/", response_class=HTMLResponse, tags=["Web UI"])
def serve_index_html():
    """Serves the primary light-themed RepoLens Web UI."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8", errors="replace"))
    return HTMLResponse(content="<h1>RepoLens API Online</h1><p>Visit <a href='/docs'>/docs</a> for Swagger UI.</p>")


@app.get("/api/health", tags=["Health"])
def health_check():
    return {
        "name": "RepoLens API",
        "status": "online",
        "version": "1.0.0",
        "active_repo": app_state.active_repo_source,
        "checkpointer": type(app_state.checkpointer).__name__,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
