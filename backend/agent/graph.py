import copy
import json
import logging
import operator
import os
import re
from typing import TypedDict, Annotated, List, Optional, Dict, Any

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from backend.agent.router import LLMRouter

logger = logging.getLogger(__name__)

# Words a human can type to accept the draft instead of requesting a revision.
APPROVAL_KEYWORDS = {"approve", "approved", "looks good", "lgtm", "yes", "ok", "okay"}

CODE_ONLY_SYSTEM_PROMPT = (
    "You are a Senior Python Engineer. Output ONLY fully implemented, executable, "
    "syntactically valid Python code. Never wrap the code in markdown fences, never "
    "return JSON, and never include commentary, plans, or explanations."
)


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    task_spec: str
    selected_tier: Optional[str]
    generated_plan_or_code: Optional[str]
    user_feedback: Optional[str]
    modified_spec: Optional[Dict[str, Any]]
    base_specs: Optional[Dict[str, Any]]
    drafts: Optional[List[Dict[str, Any]]]
    status: str


router = LLMRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _is_approval(text: str) -> bool:
    return (text or "").strip().lower() in APPROVAL_KEYWORDS


def _clean_code(raw: str) -> str:
    """Strips reasoning tags and markdown fences from a raw model response."""
    code = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.DOTALL)
    code = re.sub(r"^```python\s*", "", code, flags=re.MULTILINE)
    code = re.sub(r"^```\s*$", "", code, flags=re.MULTILINE)
    return code.strip()


def _specs_from_state(state: AgentState) -> Dict[str, Any]:
    """Resolves the project specification the CoderAgent should implement.

    Priority: explicit ``modified_spec`` -> parsed ``task_spec`` JSON ->
    a single synthesised feature built from free-form task text.
    """
    modified = state.get("modified_spec")
    if isinstance(modified, dict) and modified.get("features"):
        return copy.deepcopy(modified)

    raw = (state.get("task_spec") or "").strip()
    if raw.startswith("{"):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and parsed.get("features"):
                return copy.deepcopy(parsed)
        except json.JSONDecodeError:
            logger.warning("[Agent] task_spec looked like JSON but could not be parsed.")

    title = (raw.splitlines()[0][:60] if raw else "generated_module").strip() or "generated_module"
    return {
        "project_title": title,
        "features": [
            {
                "feature_name": title,
                "target_module": "generated_module",
                "business_rules": [raw] if raw else ["Implement the requested module."],
                "functions": [],
                "exceptions": [],
            }
        ],
        "cross_cutting_constraints": [],
    }


def _build_coder():
    """Instantiates the Phase 2 CoderAgent (Groq primary, Gemini fallback)."""
    from phase2_coder.coder import CoderAgent

    return CoderAgent(
        groq_api_key=os.getenv("GROQ_API_KEY") or os.getenv("groq_api_key") or "",
        gemini_api_key=os.getenv("GEMINI_API_KEY") or os.getenv("gemini_api_key") or "",
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY") or os.getenv("openrouter_api_key"),
    )


def _format_bundle(drafts: List[Dict[str, Any]]) -> str:
    """Renders the generated modules as a single displayable Python bundle."""
    blocks = []
    for draft in drafts:
        divider = "# " + "=" * 68
        header = (
            f"{divider}\n"
            f"# MODULE: {draft.get('target_module')}\n"
            f"# MCP SYNTAX CHECK: {draft.get('validation_msg', 'not run')}\n"
            f"{divider}"
        )
        blocks.append(f"{header}\n{draft.get('code', '')}")
    return "\n\n\n".join(blocks)


def _fallback_generation(state: AgentState, specs: Dict[str, Any]) -> str:
    """Last-resort single-shot generation through the LangChain router model."""
    model = router.get_model(state.get("selected_tier") or "fast")
    prompt = (
        f"{CODE_ONLY_SYSTEM_PROMPT}\n\n"
        f"Implement the following specification:\n{json.dumps(specs, indent=2)}"
    )
    response = model.invoke([HumanMessage(content=prompt)])
    return _clean_code(getattr(response, "content", str(response)))


def route_task_node(state: AgentState):
    """Classifies the incoming spec so the agent knows which tier to target."""
    spec = state["task_spec"]
    try:
        classification = router.route(spec)
        tier = classification.recommended_model
        reason = classification.reasoning
    except Exception as exc:  # routing is best-effort; never block generation
        logger.warning(f"[Agent] Task classification failed ({exc}); defaulting to 'fast'.")
        tier, reason = "fast", "Classification unavailable; defaulted to the fast tier."

    return {
        "selected_tier": tier,
        "status": "planning",
        "messages": [AIMessage(content=f"Routed to '{tier}' tier. Reason: {reason}")],
    }


def generate_code_node(state: AgentState):
    """Generates REAL Python modules by delegating to the Phase 2 CoderAgent.

    The CoderAgent prompt-engineers whole modules, cleans the model response, runs
    the MCP AST syntax guardrail, stages the draft and executes the MCP pytest tool.
    Human revision feedback is passed straight into the module prompt as a
    highest-priority instruction, and the ORIGINAL spec is always reused so
    repeated revisions never accumulate duplicated rules.
    """
    base_specs = state.get("base_specs") or _specs_from_state(state)
    feedback = (state.get("user_feedback") or "").strip()
    is_revision = bool(feedback) and not _is_approval(feedback)
    features = base_specs.get("features", [])

    drafts: List[Dict[str, Any]] = []
    try:
        coder = _build_coder()
        if is_revision:
            logger.info(
                f"[Agent] Applying human revision notes to {len(features)} module(s)..."
            )
            for feature in features:
                try:
                    drafts.append(
                        coder.generate_single_module(feature, revision_notes=feedback)
                    )
                except Exception as exc:
                    logger.error(
                        f"[Agent] Revision failed for "
                        f"'{feature.get('target_module')}': {exc}"
                    )
        else:
            drafts = coder.generate_drafts_from_specs(base_specs)
    except Exception as exc:
        logger.error(f"[Agent] CoderAgent generation failed: {exc}")

    drafts = [d for d in drafts if (d.get("code") or "").strip()]

    try:
        if drafts:
            bundle = _format_bundle(drafts)
            verb = "Regenerated" if is_revision else "Generated"
            note = (
                f"{verb} {len(drafts)} module draft(s) via CoderAgent + MCP "
                f"validation. Awaiting human approval."
            )
        else:
            logger.warning("[Agent] CoderAgent produced no drafts; using router fallback.")
            bundle = _fallback_generation(state, base_specs)
            note = "CoderAgent produced no drafts; returned a single-module fallback draft."
    except Exception as exc:
        logger.error(f"[Agent] Fallback generation failed: {exc}")
        bundle = f"# Agent could not generate code: {exc}"
        note = f"Generation failed: {exc}"

    return {
        "generated_plan_or_code": bundle,
        "drafts": drafts,
        "base_specs": base_specs,
        "messages": [AIMessage(content=note)],
        "status": "awaiting_approval",
    }


def process_feedback_node(state: AgentState):
    feedback = (state.get("user_feedback") or "").strip()
    if _is_approval(feedback):
        return {"status": "approved"}

    if not feedback:
        return {
            "status": "revising",
            "messages": [AIMessage(content="No feedback supplied; regenerating the draft.")],
        }

    return {
        "status": "revising",
        "messages": [HumanMessage(content=f"User feedback for revision: {feedback}")],
    }


def finalize_node(state: AgentState):
    """Commits the human-approved modules into generated_src/ via the MCP file tool."""
    from mcp_tools.file_tools import MCPToolServer
    from phase2_coder.hitl_manager import hitl_queue

    written, failures = [], []
    for draft in state.get("drafts") or []:
        try:
            path = MCPToolServer.write_module_to_disk(
                draft["target_module"], draft["code"], base_dir="generated_src"
            )
            written.append(path)
            hitl_queue.update_status(draft["draft_id"], "COMMITTED", draft["code"])
        except Exception as exc:
            failures.append(f"{draft.get('target_module')}: {exc}")

    if written:
        logger.info(f"[Agent] Committed {len(written)} approved module(s) via MCP tool.")
    if failures:
        logger.error(f"[Agent] MCP commit failures: {'; '.join(failures)}")

    summary = f"Approved and committed {len(written)} module(s) to generated_src/."
    if failures:
        summary += f" Failures: {'; '.join(failures)}"

    return {"status": "approved", "messages": [AIMessage(content=summary)]}


def decide_next_step(state: AgentState):
    if state["status"] == "approved":
        return "finalize"
    return "generate_code"


def build_agent_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("route_task", route_task_node)
    workflow.add_node("generate_code", generate_code_node)
    workflow.add_node("process_feedback", process_feedback_node)
    workflow.add_node("finalize", finalize_node)

    workflow.set_entry_point("route_task")
    workflow.add_edge("route_task", "generate_code")
    workflow.add_edge("generate_code", "process_feedback")

    workflow.add_conditional_edges(
        "process_feedback",
        decide_next_step,
        {
            "finalize": "finalize",
            "generate_code": "generate_code"
        }
    )
    workflow.add_edge("finalize", END)

    checkpointer = MemorySaver()
    return workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["process_feedback"]
    )


agent_app = build_agent_graph()
