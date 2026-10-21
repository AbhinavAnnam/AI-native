import os
import logging
import asyncio
from collections import defaultdict
from fastapi import APIRouter, HTTPException
from starlette.concurrency import run_in_threadpool

from backend.models.schemas import (
    ProjectSpecs,
    HITLApprovalPayload,
    StartExecutionRequest,
    HumanFeedbackRequest
)
from backend.agent.graph import agent_app, _is_approval
from backend.workspace import clear_generated_workspace
from backend.stages.critic import CriticError, ensure_current_review, snapshot_candidate
from backend.stages.documentation import delivery_view

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
workflow_locks = defaultdict(asyncio.Lock)


def _critic_summary(values):
    report = values.get("critic_report")
    progress = values.get("critic_progress") or {}
    pending = values.get("critic_pending_repair") or {}
    stale = False
    if report or progress or pending:
        try:
            snapshot = snapshot_candidate(values)
            fingerprint = snapshot["input_fingerprint"]
            stale = bool(report and report["input_fingerprint"] != fingerprint)
            from backend.stages.critic import pending_repair_matches
            pending = pending if pending_repair_matches(snapshot, pending) else {}
            if not pending and progress.get("input_fingerprint") != fingerprint:
                progress = {}
        except (ValueError, OSError, RuntimeError):
            stale = True
            progress, pending = {}, {}
    return {**delivery_view(values).model_dump(), "critic_report": report, "critic_stale": stale,
            "critic_repair_rounds": values.get("critic_repair_rounds", 0),
            "critic_previous_drafts": values.get("critic_previous_drafts") or [],
            "critic_pending_repair": bool(pending),
            "critic_repair_in_progress": pending.get("phase") == "repair",
            "critic_progress": {key: value for key, value in progress.items() if key != "results"}}


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
    raise HTTPException(409, "Use Agent Studio and the critic approval workflow to save reviewed files.")


# ==========================================
# 2. LangGraph Execution Engine (HITL Stream)
# ==========================================

def _draft_summaries(values: dict) -> list:
    """Return separate files, including code, for the module review cards."""
    return [
        {
            "draft_id": draft.get("draft_id"),
            "target_module": draft.get("target_module"),
            "file_path": draft.get("file_path") or draft["target_module"].replace(".", "/") + ".py",
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


def _repository_summary(values: dict):
    repository = values.get("repository")
    if not repository:
        return None
    return {key: repository[key] for key in (
        "run_id", "url", "repo_dir", "base_branch", "baseline_commit", "work_branch", "file_mappings",
    )}


def _generation_summary(values: dict):
    available = {d["target_module"] for d in values.get("drafts") or []
                 if d.get("syntax_valid") and (d.get("code") or "").strip()}
    return {"missing_modules": [f["target_module"] for f in (values.get("base_specs") or {}).get("features", [])
                                if f["target_module"] not in available]}


@router.post("/start")
async def start_agent_workflow(req: StartExecutionRequest):
    config = {"configurable": {"thread_id": req.thread_id}}
    if agent_app.get_state(config).values:
        raise HTTPException(status_code=409, detail="This thread already started. Create a new thread for a new run.")
    if not req.task_spec.strip():
        raise HTTPException(status_code=400, detail="Task specification must not be empty.")
    if not req.repository:
        _reset_generated_workspace()
    initial_state = {
        "task_spec": req.task_spec,
        "messages": [],
        "user_feedback": None,
        "modified_spec": None,
        "repository_request": req.repository.model_dump() if req.repository else None,
        "repository": None,
        "status": "started"
    }

    await run_in_threadpool(agent_app.invoke, initial_state, config)

    state = agent_app.get_state(config)
    return {
        "thread_id": req.thread_id,
        "next_step": state.next,
        "generated_output": state.values.get("generated_plan_or_code"),
        "modules": _draft_summaries(state.values),
        "module_groups": state.values.get("module_groups") or {},
        "status": state.values.get("status"),
        "repository": _repository_summary(state.values),
        "error": state.values.get("error"),
        "candidate_version": state.values.get("candidate_version"),
        **_generation_summary(state.values),
        **_critic_summary(state.values),
    }


@router.get("/state/{thread_id}")
async def get_agent_state(thread_id: str):
    """Refresh review files without resuming generation or approving them."""
    state = agent_app.get_state({"configurable": {"thread_id": thread_id}})
    if not state.values:
        raise HTTPException(status_code=404, detail="This review is no longer available on the backend. Download any visible drafts before starting a New Thread.")
    return {
        "thread_id": thread_id,
        "is_completed": not state.next,
        "status": state.values.get("status"),
        "latest_output": state.values.get("generated_plan_or_code"),
        "modules": _draft_summaries(state.values),
        "module_groups": state.values.get("module_groups") or {},
        "repository": _repository_summary(state.values),
        "error": state.values.get("error"),
        "candidate_version": state.values.get("candidate_version"),
        **_generation_summary(state.values),
        **_critic_summary(state.values),
    }


@router.post("/feedback")
async def submit_human_feedback(req: HumanFeedbackRequest):
    async with workflow_locks[req.thread_id]:
        return await _submit_human_feedback(req)


async def _submit_human_feedback(req: HumanFeedbackRequest):
    config = {"configurable": {"thread_id": req.thread_id}}
    state = agent_app.get_state(config)
    if not state.values:
        raise HTTPException(status_code=404, detail="This review is no longer available on the backend. Download any visible drafts before starting a New Thread.")
    if state.next != ("process_feedback",):
        raise HTTPException(status_code=409, detail="This thread is not waiting for feedback.")
    if req.modified_spec is not None and state.values.get("repository"):
        modules = [f.get("target_module") for f in req.modified_spec.get("features", [])]
        mapped = state.values["repository"]["file_mappings"]
        if len(set(modules)) != len(modules) or set(modules) != set(mapped):
            raise HTTPException(status_code=400, detail="Updated specifications must retain the repository's mapped modules. Start a new thread to change file mappings.")
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
        try:
            snapshot = ensure_current_review(state.values, req.input_fingerprint)
            if (state.values.get("critic_pending_repair") or {}).get("origin_fingerprint") == snapshot["input_fingerprint"]:
                raise CriticError("Resume the pending repaired candidate's critic review before approval.")
        except (CriticError, OSError, ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    # An empty submission must leave the existing interrupt untouched.
    if not req.user_feedback.strip() and not req.modified_spec:
        return {
            "thread_id": req.thread_id, "is_completed": False,
            "status": state.values.get("status"),
            "latest_output": state.values.get("generated_plan_or_code"),
            "modules": _draft_summaries(state.values),
            "repository": _repository_summary(state.values),
            "candidate_version": state.values.get("candidate_version"),
            **_generation_summary(state.values),
            **_critic_summary(state.values),
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

    await run_in_threadpool(agent_app.invoke, None, config)

    state = agent_app.get_state(config)
    return {
        "thread_id": req.thread_id,
        "is_completed": len(state.next) == 0,
        "status": state.values.get("status"),
        "latest_output": state.values.get("generated_plan_or_code"),
        "modules": _draft_summaries(state.values),
        "repository": _repository_summary(state.values),
        "candidate_version": state.values.get("candidate_version"),
        **_generation_summary(state.values),
        **_critic_summary(state.values),
    }
