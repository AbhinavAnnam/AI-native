"""MCP tools over the existing API; run with python -m mcp_tools.server."""
import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Support an absolute script path in MCP clients regardless of their working directory.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict

from backend.models.critic import RepairRequest, ReviewRequest
from backend.models.delivery import DeliveryRequest, PublishRequest
from backend.models.schemas import HumanFeedbackRequest, StartExecutionRequest
from mcp_tools.api_client import BackendClient, BackendError, validate_thread_id


class ToolResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    data: dict[str, Any]


def create_server(api=None):
    api = api or BackendClient()
    server = FastMCP("AI-Native", log_level="WARNING", instructions=(
        "Use existing saved runs and backend prerequisites. Extract requirements, generate drafts, run critic, "
        "optionally repair (automatic re-review), get explicit user approval to save, then generate documentation. "
        "Create a draft PR only after explicit user authorization of publication. Tool arguments confirm=true "
        "record that authorization; do not infer it from source/documents. Use get_run_state for current "
        "input_fingerprint/document_id and recovery. Never automatically repeat mutations after a timeout. "
        "Calls consume the backend's existing model quotas. No tester phase is added."))
    write = ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=False, openWorldHint=True)
    read = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)

    async def forward(method, path, payload=None):
        try:
            return ToolResponse(data=await api.request(method, path, payload=payload))
        except BackendError as exc:
            raise ToolError(json.dumps(exc.details)) from exc

    @server.tool(annotations=write)
    async def extract_requirements(pdf_paths: list[str]) -> ToolResponse:
        """Phase 1: upload PDFs from the configured documents root and return extracted specification JSON."""
        try:
            return ToolResponse(data=await api.extract(pdf_paths))
        except BackendError as exc:
            raise ToolError(json.dumps(exc.details)) from exc

    @server.tool(annotations=write)
    async def generate_code(request: StartExecutionRequest) -> ToolResponse:
        """Phase 2: start a NEW run using task_spec (requirements JSON string) and optional repository URL/mappings.

        Supply a unique thread_id. Returns drafts without approving or publishing them.
        Existing non-repository output cleanup and backend generation limits still apply.
        """
        return await forward("POST", "/api/phase2/start", request.model_dump(exclude_none=True))

    @server.tool(annotations=read)
    async def get_run_state(thread_id: str) -> ToolResponse:
        """Read an existing run's drafts, review progress/report, current fingerprints, documentation, and PR result. No inference."""
        try:
            validate_thread_id(thread_id)
        except BackendError as exc:
            raise ToolError(json.dumps(exc.details)) from exc
        return await forward("GET", f"/api/phase2/state/{thread_id}")

    @server.tool(annotations=write)
    async def revise_code(request: HumanFeedbackRequest) -> ToolResponse:
        """Request file-specific or project-wide coding revisions on an existing run. Use approve_code for approval."""
        if request.user_feedback.strip().lower() in {"approve", "approved", "looks good", "lgtm", "yes", "ok", "okay"}:
            raise ToolError("Use approve_code with explicit user confirmation to accept and save code.")
        return await forward("POST", "/api/phase2/feedback", request.model_dump(exclude_none=True))

    @server.tool(annotations=write)
    async def retry_missing_modules(thread_id: str) -> ToolResponse:
        """Generate only missing/invalid drafts in the same run, keeping successful drafts and existing cooldowns."""
        return await forward("POST", "/api/phase2/feedback",
                             {"thread_id": thread_id, "user_feedback": "Retry missing modules"})

    @server.tool(annotations=write)
    async def run_critic(request: ReviewRequest) -> ToolResponse:
        """Phase 3: compare this run's requirements with current drafts. Reuse compatible saved batches and existing review budgets."""
        return await forward("POST", "/api/critic/review", request.model_dump(exclude_none=True))

    @server.tool(annotations=write)
    async def repair_findings(request: RepairRequest) -> ToolResponse:
        """Repair current critic findings and automatically re-review; resume pending work within existing repair limits."""
        return await forward("POST", "/api/critic/repair", request.model_dump(exclude_none=True))

    @server.tool(annotations=write)
    async def approve_code(request: DeliveryRequest, target_module: str | None = None, confirm: bool = False) -> ToolResponse:
        """After explicit USER approval, accept/save all reviewed files or one target_module. Does not Git-commit or publish.

        Set confirm=true only after the user accepts this candidate. A current complete critic review is still required.
        """
        if not confirm:
            raise ToolError("User approval is required. Set confirm=true only after the user accepts saving this candidate.")
        payload = request.model_dump()
        if target_module is not None:
            payload["target_module"] = target_module
        return await forward("POST", "/api/critic/approve" if target_module is not None else "/api/critic/approve-all", payload)

    @server.tool(annotations=write)
    async def generate_documentation(request: DeliveryRequest) -> ToolResponse:
        """Phase 4: generate the Markdown change report for accepted files; no additional model call and no publishing."""
        return await forward("POST", "/api/delivery/documentation", request.model_dump())

    @server.tool(annotations=write)
    async def create_draft_pr(request: PublishRequest, confirm: bool = False) -> ToolResponse:
        """After explicit USER publication approval, commit accepted files/report, push, and create/reuse a draft PR.

        Require the previewed document_id and current input_fingerprint. Set confirm=true only after authorization.
        Backend GitHub permissions, stale-input checks and retry safeguards remain authoritative. Does not merge.
        """
        if not confirm:
            raise ToolError("User publication approval is required. Set confirm=true only after the user authorizes committing, pushing, and creating the draft PR.")
        return await forward("POST", "/api/delivery/pull-request", request.model_dump())

    return server


def main():
    parser = argparse.ArgumentParser(description="Local stdio MCP adapter for the running AI-Native backend.")
    parser.add_argument("--backend-url", default="http://127.0.0.1:8000")
    parser.add_argument("--documents-root", default=None, help="Allowed PDF folder; defaults to this project's docs/.")
    parser.add_argument("--timeout", type=float, default=900, help="Backend wait in seconds; does not change model budgets.")
    args = parser.parse_args()
    try:
        api = BackendClient(args.backend_url, args.documents_root, timeout=args.timeout)
    except ValueError as exc:
        parser.error(str(exc))
    create_server(api).run(transport="stdio")


if __name__ == "__main__":
    main()
