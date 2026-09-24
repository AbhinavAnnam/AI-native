import os
import logging
from fastapi import APIRouter, HTTPException

from backend.models.schemas import (
    ProjectSpecs,
    HITLApprovalPayload,
    StartExecutionRequest,
    HumanFeedbackRequest
)
from backend.agent.graph import agent_app

# Fallback imports to support different working directory executions
try:
    from phase2_coder.coder import CoderAgent
    from phase2_coder.hitl_manager import hitl_queue
    from mcp_tools.file_tools import MCPToolServer
except ImportError:
    import sys
    from pathlib import Path

    root_dir = Path(__file__).resolve().parents[2]
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))

    from phase2_coder.coder import CoderAgent
    from phase2_coder.hitl_manager import hitl_queue
    from mcp_tools.file_tools import MCPToolServer

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/phase2", tags=["Phase 2 Agent & Coder"])


# ==========================================
# 1. Coder Agent Draft Generation & MCP Tools
# ==========================================

@router.post("/generate-drafts")
async def generate_drafts(specs: ProjectSpecs):
    groq_key = os.getenv("GROQ_API_KEY") or os.getenv("groq_api_key")
    gemini_key = os.getenv("GEMINI_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("openrouter_api_key")

    if not gemini_key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is required in .env file.",
        )

    if not groq_key and not openrouter_key:
        raise HTTPException(
            status_code=500,
            detail="At least one primary key (GROQ_API_KEY or OPENROUTER_API_KEY) must be provided in .env.",
        )

    logger.info("Starting Phase 2 Coder Agent draft generation...")

    try:
        from main import clear_generated_workspace
        clear_generated_workspace("generated_src")
    except ImportError:
        try:
            from backend.main import clear_generated_workspace
            clear_generated_workspace("generated_src")
        except Exception as err:
            logger.warning(f"Could not clear workspace automatically: {err}")

    drafts = CoderAgent(
        groq_api_key=groq_key or "",
        gemini_api_key=gemini_key,
        openrouter_api_key=openrouter_key,
    ).generate_drafts_from_specs(specs.model_dump())

    return {"status": "success", "pending_reviews": drafts}


@router.post("/approve-and-commit")
async def approve_and_commit(payload: HITLApprovalPayload):
    if payload.action == "reject":
        hitl_queue.update_status(payload.draft_id, "REJECTED")
        return {"status": "rejected", "module": payload.target_module}
        
    try:
        written_file_path = MCPToolServer.write_module_to_disk(
            payload.target_module, payload.approved_code
        )
        hitl_queue.update_status(
            payload.draft_id, "COMMITTED", payload.approved_code
        )
        return {
            "status": "committed",
            "file_path": written_file_path,
            "module": payload.target_module,
        }
    except Exception as exc:
        logger.error(f"MCP Write Error: {str(exc)}")
        raise HTTPException(status_code=400, detail=str(exc))


# ==========================================
# 2. LangGraph Execution Engine (HITL Stream)
# ==========================================

def _draft_summaries(values: dict) -> list:
    """Lightweight per-module view returned to the UI for the approval cards."""
    return [
        {
            "draft_id": draft.get("draft_id"),
            "target_module": draft.get("target_module"),
            "validation_msg": draft.get("validation_msg"),
            "syntax_valid": draft.get("syntax_valid"),
            "tests_passed": draft.get("tests_passed"),
        }
        for draft in (values.get("drafts") or [])
    ]


@router.post("/start")
async def start_agent_workflow(req: StartExecutionRequest):
    config = {"configurable": {"thread_id": req.thread_id}}
    initial_state = {
        "task_spec": req.task_spec,
        "messages": [],
        "user_feedback": None,
        "modified_spec": None,
        "status": "started"
    }

    async for _ in agent_app.astream(initial_state, config):
        pass

    state = agent_app.get_state(config)
    return {
        "thread_id": req.thread_id,
        "next_step": state.next,
        "generated_output": state.values.get("generated_plan_or_code"),
        "modules": _draft_summaries(state.values),
        "status": state.values.get("status")
    }


@router.post("/feedback")
async def submit_human_feedback(req: HumanFeedbackRequest):
    config = {"configurable": {"thread_id": req.thread_id}}

    agent_app.update_state(
        config,
        {
            "user_feedback": req.user_feedback,
            "modified_spec": req.modified_spec
        },
        as_node="generate_code"
    )

    async for _ in agent_app.astream(None, config):
        pass

    state = agent_app.get_state(config)
    return {
        "thread_id": req.thread_id,
        "is_completed": len(state.next) == 0,
        "status": state.values.get("status"),
        "latest_output": state.values.get("generated_plan_or_code"),
        "modules": _draft_summaries(state.values)
    }
