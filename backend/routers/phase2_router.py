import os
import logging
from fastapi import APIRouter, HTTPException

from backend.models.schemas import (
    ProjectSpecs,
    HITLApprovalPayload,
    StartExecutionRequest,
    HumanFeedbackRequest
)
from backend.agent.graph import agent_app, _is_approval
from backend.workspace import clear_generated_workspace

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


def _reset_generated_workspace():
    try:
        clear_generated_workspace()
    except (OSError, ValueError) as exc:
        logger.error("Could not clear generated_src for a new run: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Could not clear generated_src. Release any locked files and retry; generation has not started.",
        ) from exc
    hitl_queue.clear()


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

    _reset_generated_workspace()

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
    """Return separate files, including code, for the module review cards."""
    return [
        {
            "draft_id": draft.get("draft_id"),
            "target_module": draft.get("target_module"),
            "file_path": draft["target_module"].replace(".", "/") + ".py",
            "feature_name": draft.get("feature_name"),
            "code": draft.get("code", ""),
            "validation_msg": draft.get("validation_msg"),
            "syntax_valid": draft.get("syntax_valid"),
            "tests_passed": draft.get("tests_passed"),
            "test_output": draft.get("test_output", ""),
            "status": draft.get("status", "AWAITING_HUMAN_REVIEW"),
            "commit_error": draft.get("commit_error"),
        }
        for draft in (values.get("drafts") or [])
    ]


@router.post("/start")
async def start_agent_workflow(req: StartExecutionRequest):
    config = {"configurable": {"thread_id": req.thread_id}}
    if agent_app.get_state(config).values:
        raise HTTPException(status_code=409, detail="This thread already started. Create a new thread for a new run.")
    if not req.task_spec.strip():
        raise HTTPException(status_code=400, detail="Task specification must not be empty.")
    _reset_generated_workspace()
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


@router.get("/state/{thread_id}")
async def get_agent_state(thread_id: str):
    """Refresh review files without resuming generation or approving them."""
    state = agent_app.get_state({"configurable": {"thread_id": thread_id}})
    if not state.values:
        raise HTTPException(status_code=404, detail="Agent thread not found. Start a new thread if the backend restarted.")
    return {
        "thread_id": thread_id,
        "is_completed": not state.next,
        "status": state.values.get("status"),
        "latest_output": state.values.get("generated_plan_or_code"),
        "modules": _draft_summaries(state.values),
    }


@router.post("/feedback")
async def submit_human_feedback(req: HumanFeedbackRequest):
    config = {"configurable": {"thread_id": req.thread_id}}
    state = agent_app.get_state(config)
    if not state.values:
        raise HTTPException(status_code=404, detail="Agent thread not found.")
    if state.next != ("process_feedback",):
        raise HTTPException(status_code=409, detail="This thread is not waiting for feedback.")
    if req.target_module is not None:
        targets = {f["target_module"] for f in state.values["base_specs"]["features"]}
        if req.target_module not in targets:
            raise HTTPException(status_code=400, detail="Unknown target module.")
        if req.modified_spec is not None:
            raise HTTPException(status_code=400, detail="Use project feedback to change the specification.")
    if _is_approval(req.user_feedback):
        if req.target_module is None:
            raise HTTPException(status_code=400, detail="Approve files individually using each file's approval button.")
        draft = next((d for d in state.values.get("drafts") or [] if d["target_module"] == req.target_module), None)
        if not draft or not draft.get("syntax_valid") or not (draft.get("code") or "").strip():
            raise HTTPException(status_code=400, detail="Generate a valid draft for this file before approval.")
        if draft.get("status") == "COMMITTED":
            raise HTTPException(status_code=409, detail="This file is already approved and saved.")

    # An empty submission must leave the existing interrupt untouched.
    if not req.user_feedback.strip() and not req.modified_spec:
        return {
            "thread_id": req.thread_id, "is_completed": False,
            "status": state.values.get("status"),
            "latest_output": state.values.get("generated_plan_or_code"),
            "modules": _draft_summaries(state.values),
        }

    agent_app.update_state(
        config,
        {
            "user_feedback": req.user_feedback.strip() or "Implement the updated specification.",
            "modified_spec": req.modified_spec,
            "revision_target": req.target_module,
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
