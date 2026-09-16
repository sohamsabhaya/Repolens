"""
Conversational Agent & HITL Chat Routes
"""

from fastapi import APIRouter, HTTPException
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from src.agent.checkpoint import create_thread_config, generate_thread_id
from src.api.schemas import ChatRequest, ChatResponse, ResumeRequest

router = APIRouter(prefix="/api", tags=["Chat & HITL"])


@router.post("/chat", response_model=ChatResponse)
def chat_with_codebase(req: ChatRequest):
    """
    Executes a multi-turn agentic query against the ingested codebase.
    Detects if LangGraph interrupt() is triggered for human disambiguation.
    """
    from src.api.main import app_state

    if not app_state.agent_app:
        raise HTTPException(status_code=400, detail="No repository has been ingested yet. Call /api/ingest first.")

    thread_id = req.thread_id or generate_thread_id()
    config = create_thread_config(thread_id)

    initial_input = {
        "messages": [HumanMessage(content=req.query)],
        "retrieved_docs": [],
        "citations": [],
    }

    try:
        final_state = app_state.agent_app.invoke(initial_input, config)
        
        # Check if graph paused due to interrupt()
        graph_state = app_state.agent_app.get_state(config)
        is_interrupted = bool(graph_state.next)
        interrupt_payload = None
        
        if is_interrupted and graph_state.tasks:
            for task in graph_state.tasks:
                if hasattr(task, "interrupts") and task.interrupts:
                    interrupt_payload = task.interrupts[0].value
                    break

        response_text = ""
        if final_state.get("messages"):
            response_text = final_state["messages"][-1].content

        return ChatResponse(
            thread_id=thread_id,
            response=response_text,
            intent=final_state.get("intent"),
            citations=final_state.get("citations", []),
            is_interrupted=is_interrupted,
            interrupt_payload=interrupt_payload,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat execution failed: {str(e)}")


@router.post("/resume", response_model=ChatResponse)
def resume_interrupted_session(req: ResumeRequest):
    """
    Resumes a paused Human-in-the-Loop session with user-selected disambiguation.
    """
    from src.api.main import app_state

    if not app_state.agent_app:
        raise HTTPException(status_code=400, detail="Agent graph not initialized.")

    config = create_thread_config(req.thread_id)
    try:
        resumed_state = app_state.agent_app.invoke(Command(resume=req.resume_value), config)
        
        response_text = ""
        if resumed_state.get("messages"):
            response_text = resumed_state["messages"][-1].content

        return ChatResponse(
            thread_id=req.thread_id,
            response=response_text,
            intent=resumed_state.get("intent"),
            citations=resumed_state.get("citations", []),
            is_interrupted=False,
            interrupt_payload=None,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to resume session: {str(e)}")
