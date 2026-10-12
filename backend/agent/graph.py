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
    revision_target: Optional[str]
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
    """Last-resort generation for ONE module through the LangChain router."""
    model = router.get_model(state.get("selected_tier") or "fast")
    module = specs["features"][0]["target_module"]
    current_code = next((d["code"] for d in state.get("drafts") or [] if d["target_module"] == module), "")
    prompt = (
        f"{CODE_ONLY_SYSTEM_PROMPT}\n\n"
        f"Implement only the module in this specification:\n{json.dumps(specs, indent=2)}\n"
        f"Human revision notes: {state.get('user_feedback') or 'None'}\n"
        f"Current code (preserve existing behavior except for requested changes):\n{current_code}"
    )
    response = model.invoke([HumanMessage(content=prompt)])
    return _clean_code(getattr(response, "content", str(response)))


def route_task_node(state: AgentState):
    """This workflow always generates code; no LLM classification is needed."""
    specs = _specs_from_state(state)
    tier = "heavy" if len(specs.get("features", [])) > 1 else "fast"

    return {
        "selected_tier": tier,
        "base_specs": specs,
        "status": "planning",
        "messages": [AIMessage(content=f"Generating modules via CoderAgent; fallback tier: '{tier}'.")],
    }


def generate_code_node(state: AgentState):
    """Generates REAL Python modules by delegating to the Phase 2 CoderAgent.

    The CoderAgent prompt-engineers whole modules, cleans the model response, runs
    the MCP AST syntax guardrail, stages the draft and executes the MCP pytest tool.
    Human revision feedback is passed straight into the module prompt as a
    highest-priority instruction, and the ORIGINAL spec is always reused so
    repeated revisions never accumulate duplicated rules.
    """
    base_specs = (
        _specs_from_state(state) if state.get("modified_spec")
        else state.get("base_specs") or _specs_from_state(state)
    )
    feedback = (state.get("user_feedback") or "").strip()
    is_revision = bool(feedback) and not _is_approval(feedback)
    features = base_specs.get("features", [])
    target = state.get("revision_target") if is_revision else None
    selected_features = [f for f in features if f["target_module"] == target] if target else features
    current_codes = {d["target_module"]: d["code"] for d in state.get("drafts") or []}

    drafts: List[Dict[str, Any]] = []
    try:
        coder = _build_coder()
        if target:
            drafts = [coder.generate_single_module(
                selected_features[0], revision_notes=feedback,
                current_code=current_codes.get(target, ""),
            )]
        else:
            drafts = coder.generate_drafts_from_specs(
                base_specs, revision_notes=feedback if is_revision else "",
                current_codes=current_codes if is_revision else None,
            )
    except Exception as exc:
        logger.error(f"[Agent] CoderAgent generation failed: {exc}")

    # Every newly generated version needs its own approval, including revisions
    # of a file that has previously been saved.
    drafts = [
        {**d, "status": "AWAITING_HUMAN_REVIEW", "commit_error": None}
        for d in drafts if (d.get("code") or "").strip()
    ]

    # Preserve the JSON module boundaries even if the primary coder fails.
    from mcp_tools.file_tools import MCPToolServer

    by_module = {draft["target_module"]: draft for draft in drafts}
    if target:
        by_module.update({d["target_module"]: d for d in state.get("drafts") or [] if d["target_module"] != target})
    failures = []
    for feature in selected_features:
        module = feature["target_module"]
        if module in by_module:
            continue
        try:
            module_specs = {**base_specs, "features": [feature]}
            code = _fallback_generation(state, module_specs)
            valid, message = MCPToolServer.validate_python_syntax(code)
            if not code.strip() or not valid:
                raise ValueError(message if code.strip() else "Empty model response")
            by_module[module] = {
                "draft_id": f"draft_{module.replace('.', '_')}",
                "feature_name": feature.get("feature_name"),
                "target_module": module, "code": code,
                "syntax_valid": valid, "validation_msg": message,
                "tests_passed": False, "test_output": "Not run (router fallback).",
                "status": "AWAITING_HUMAN_REVIEW", "commit_error": None,
            }
        except Exception as exc:
            logger.error("[Agent] Fallback failed for %s: %s", module, exc)
            failures.append(module)

    drafts = [by_module[f["target_module"]] for f in features if f["target_module"] in by_module]
    bundle = _format_bundle(drafts)
    note = (
        f"Updated {target}. Awaiting human approval." if target else
        f"{'Regenerated' if is_revision else 'Generated'} {len(drafts)} module draft(s). Awaiting human approval."
    )
    if failures:
        note = f"Generation failed for: {', '.join(failures)}. Submit revision notes to retry."
    if not bundle:
        bundle = note

    return {
        "generated_plan_or_code": bundle,
        "drafts": drafts,
        "base_specs": base_specs,
        "user_feedback": None,
        "revision_target": None,
        "modified_spec": None,
        "messages": [AIMessage(content=note)],
        "status": "awaiting_approval",
    }


def process_feedback_node(state: AgentState):
    feedback = (state.get("user_feedback") or "").strip()
    if _is_approval(feedback):
        return {"status": "approved"}

    if not feedback:
        return {
            "status": "awaiting_approval",
            "messages": [AIMessage(content="No feedback supplied; keeping the current draft.")],
        }

    return {
        "status": "revising",
        "messages": [HumanMessage(content=f"User feedback for revision: {feedback}")],
    }


def finalize_node(state: AgentState):
    """Save only the selected file and keep the review open for pending files."""
    from mcp_tools.file_tools import MCPToolServer
    from phase2_coder.hitl_manager import hitl_queue

    target = state.get("revision_target")
    drafts = copy.deepcopy(state.get("drafts") or [])
    draft = next((d for d in drafts if d["target_module"] == target), None)
    failed = False
    if draft is None:
        failed = True
        summary = "Select an available draft file to approve."
    elif draft.get("status") == "COMMITTED":
        summary = f"{target} is already saved."
    else:
        try:
            MCPToolServer.write_module_to_disk(
                draft["target_module"], draft["code"], base_dir="generated_src"
            )
            draft["status"] = "COMMITTED"
            draft["commit_error"] = None
            hitl_queue.update_status(draft["draft_id"], "COMMITTED", draft["code"])
            summary = f"Approved and saved {target} to generated_src/."
        except Exception as exc:
            failed = True
            draft["status"] = "COMMIT_FAILED"
            draft["commit_error"] = str(exc)
            summary = f"Could not save {target}: {exc}. You can retry approval."
            logger.error("[Agent] %s", summary)

    expected = {f["target_module"] for f in state["base_specs"]["features"]}
    committed = {d["target_module"] for d in drafts if d.get("status") == "COMMITTED"}
    complete = bool(expected) and committed == expected
    return {
        "drafts": drafts,
        "status": "commit_failed" if failed else "approved" if complete else "awaiting_approval",
        "user_feedback": None,
        "revision_target": None,
        "messages": [AIMessage(content=summary)],
    }


def decide_next_step(state: AgentState):
    if state["status"] == "approved":
        return "finalize"
    if state["status"] == "revising":
        return "generate_code"
    return END


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
            "generate_code": "generate_code",
            END: END,
        }
    )
    workflow.add_conditional_edges(
        "finalize",
        lambda state: END if state["status"] == "approved" else "process_feedback",
        {END: END, "process_feedback": "process_feedback"},
    )

    checkpointer = MemorySaver()
    return workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["process_feedback"]
    )


agent_app = build_agent_graph()
