import os
import logging
from fastapi import APIRouter, HTTPException

from models.schemas import ProjectSpecs, HITLApprovalPayload
from phase2_coder.coder import CoderAgent
from phase2_coder.hitl_manager import hitl_queue
from mcp_tools.file_tools import MCPToolServer

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/phase2", tags=["Phase 2 Coder"])


@router.post("/generate-drafts")
async def generate_drafts(specs: ProjectSpecs):
    groq_key = os.getenv("GROQ_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")

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

    from main import clear_generated_workspace
    clear_generated_workspace("generated_src")

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