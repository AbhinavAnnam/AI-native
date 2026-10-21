"""Thin API adapter; core review/repair operations live in backend.stages.critic."""
import copy
import logging
import math
import os
import time

from fastapi import APIRouter, HTTPException
from starlette.concurrency import run_in_threadpool

from backend.models.critic import ApprovalRequest, CriticReport, RepairRequest, ReviewRequest
from backend.models.schemas import HumanFeedbackRequest
from backend.models.delivery import DeliveryRequest
from backend.agent import graph
from backend.routers import phase2_router as workflow
from backend.stages import critic
from backend.tools.git_provider import RepositoryError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/critic", tags=["Critic"])


def _state(thread_id):
    config = {"configurable": {"thread_id": thread_id}}
    state = workflow.agent_app.get_state(config)
    if not state.values:
        raise HTTPException(404, "Agent thread not found. Start an Agent Studio run first.")
    return config, state


def _persist(config, state, values):
    node = "finalize" if not state.next and state.values.get("status") == "approved" else "generate_code"
    workflow.agent_app.update_state(config, values, as_node=node)


def _discard_empty_repair(config, state):
    pending = workflow.agent_app.get_state(config).values.get("critic_pending_repair") or {}
    if pending.get("phase") == "repair" and not pending.get("completed_modules"):
        _persist(config, state, {"critic_pending_repair": None})


def _operate_once(request, repair=False):
    config, state = _state(request.thread_id)
    try:
        snapshot = critic.snapshot_candidate(state.values)
        candidate = copy.deepcopy(state.values)
        pending = candidate.get("critic_pending_repair") or {}
        if pending and not critic.pending_repair_matches(snapshot, pending):
            pending = {}
            _persist(config, state, {"critic_pending_repair": None})
        elif pending and pending.get("origin_fingerprint") != snapshot["input_fingerprint"] and pending.get("phase") == "repair":
            # Keep generated files, but stop repairs based on an older policy.
            # Review the staged candidate before acting on any more findings.
            pending.update(phase="review")
            _persist(config, state, {"critic_pending_repair": pending})
        finish_repair = repair or bool(pending)
        if repair and not pending:
            critic.ensure_current_review(candidate, request.input_fingerprint)
            report = CriticReport.model_validate(candidate["critic_report"])
            rounds = candidate.get("critic_repair_rounds", 0)
            if rounds >= critic.MAX_REPAIR_ROUNDS:
                raise critic.CriticError("Two repair rounds have been used. Use coder feedback for further revisions.")
            # Count attempts before external calls, so failed requests cannot evade the budget.
            _persist(config, state, {"critic_repair_rounds": rounds + 1})
            pending = {"origin_fingerprint": snapshot["input_fingerprint"],
                       "drafts": copy.deepcopy(state.values["drafts"]), "previous_drafts": state.values["drafts"],
                       "rounds": rounds + 1, "phase": "repair", "completed_modules": [],
                       "review_seed": copy.deepcopy(state.values.get("critic_progress") or {})}
            _persist(config, state, {"critic_pending_repair": pending})
        if pending and pending.get("phase") == "repair":
            critic.ensure_current_review(candidate)
            report = CriticReport.model_validate(candidate["critic_report"])
            def save_repair(value):
                pending.update(value)
                _persist(config, state, {"critic_pending_repair": copy.deepcopy(pending)})
            drafts = critic.repair_candidate(candidate, snapshot, report,
                repairer=critic.bounded_repairer(graph._build_coder()), progress=pending, on_progress=save_repair)
            pending.update(drafts=drafts, phase="review")
            _persist(config, state, {"critic_pending_repair": pending})
        if pending:
            candidate.update(drafts=pending["drafts"], critic_previous_drafts=pending["previous_drafts"],
                             critic_repair_rounds=pending["rounds"], status="awaiting_approval")
            candidate["candidate_version"] = critic.candidate_version(candidate)
        target_snapshot = critic.snapshot_candidate(candidate)
        progress = workflow.agent_app.get_state(config).values.get("critic_progress") or {}
        if pending and not critic._compatible_progress(target_snapshot, progress):
            progress = critic.reuse_review_progress(snapshot, target_snapshot, pending.get("review_seed") or {})
        if (progress.get("input_fingerprint") == target_snapshot["input_fingerprint"] and
            (progress.get("retry_at") or 0) > time.time()):
            raise critic.CriticPaused("Provider quota is still resetting. Saved batches will resume when ready.", progress["retry_at"] - time.time())
        def save_progress(value):
            _persist(config, state, {"critic_progress": value})
        report = critic.review_candidate(target_snapshot, cached=candidate.get("critic_report"),
                                         progress=progress, on_progress=save_progress, max_batches=request.max_batches)
        latest = workflow.agent_app.get_state(config)
        if critic.snapshot_candidate(latest.values)["input_fingerprint"] != snapshot["input_fingerprint"]:
            raise HTTPException(409, "Candidate or repository changed during review. Refresh and run Critic again.")
        updates = {"critic_report": report.model_dump(), "candidate_version": report.candidate_version}
        if finish_repair:
            updates.update({key: candidate[key] for key in (
                "drafts", "critic_repair_rounds", "critic_previous_drafts", "status", "candidate_version",
            )})
            updates["generated_plan_or_code"] = "Repaired drafts are ready for review in the critic phase."
            updates["critic_pending_repair"] = None
            # An older completed run becomes a pending review after repairs.
            workflow.agent_app.update_state(config, updates, as_node="generate_code")
        else:
            _persist(config, state, updates)
    except HTTPException:
        raise
    except critic.ProviderResponseError as exc:
        raise HTTPException(422, {"kind": "invalid_model_response", "message": str(exc),
                                  "issues": exc.issues, "retryable": False}) from exc
    except critic.ProviderBillingError as exc:
        _discard_empty_repair(config, state)
        raise HTTPException(402, {"kind": "provider_billing", "message": str(exc), "retryable": False}) from exc
    except critic.ProviderUnavailableError as exc:
        raise HTTPException(503, {"kind": "provider_unavailable", "message": str(exc), "retryable": True},
            headers={"Retry-After": str(exc.retry_after)}) from exc
    except (critic.CriticPaused, critic.ProviderRateLimitError) as exc:
        progress = workflow.agent_app.get_state(config).values.get("critic_progress") or {}
        raise HTTPException(429, {"message": str(exc), "completed": progress.get("completed", 0),
                                  "total": progress.get("total", 0), "retry_after": math.ceil(exc.retry_after)},
                            headers={"Retry-After": str(math.ceil(exc.retry_after))}) from exc
    except critic.ReviewTimeLimitError as exc:
        _discard_empty_repair(config, state)
        raise HTTPException(409, "Repair time budget reached. Completed files are saved; continue the repair.") from exc
    except (critic.CriticError, RepositoryError, OSError) as exc:
        _discard_empty_repair(config, state)
        # No arbitrary provider text/source content is sent to the client or log.
        detail = str(exc) if isinstance(exc, critic.CriticError) else "Review inputs could not be read or validated. Check the local repository workspace."
        raise HTTPException(409, detail) from exc
    except Exception as exc:
        _discard_empty_repair(config, state)
        logger.warning("Critic operation failed (%s)", type(exc).__name__)
        raise HTTPException(502, "The available models could not return a valid critic report. Your code and completed review work are saved. Try Run Critic again.") from exc


def _operate(request, repair=False):
    """One user action has one bounded budget, including a short quota recovery."""
    deadline = time.monotonic() + min(660, max(1, float(os.getenv("CRITIC_OPERATION_TIMEOUT_SECONDS", "600"))))
    token = critic.critic_deadline.set(deadline)
    try:
        for attempt in range(2):
            try:
                return _operate_once(request, repair)
            except HTTPException as exc:
                delay = float((exc.headers or {}).get("Retry-After", "0"))
                if (exc.status_code != 429 or attempt or not 0 < delay <= 20 or
                        time.monotonic() + delay + 5 >= deadline):
                    raise
                logger.info("Critic waiting %.1f seconds before resuming saved work", delay)
                time.sleep(delay)
    finally:
        critic.critic_deadline.reset(token)


@router.post("/review")
async def review(request: ReviewRequest):
    async with workflow.workflow_locks[request.thread_id]:
        await run_in_threadpool(_operate, request)
        return await workflow.get_agent_state(request.thread_id)


@router.post("/repair")
async def repair(request: RepairRequest):
    async with workflow.workflow_locks[request.thread_id]:
        await run_in_threadpool(_operate, request, True)
        return await workflow.get_agent_state(request.thread_id)


@router.post("/approve")
async def approve(request: ApprovalRequest):
    return await workflow.submit_human_feedback(HumanFeedbackRequest(
        thread_id=request.thread_id, target_module=request.target_module,
        user_feedback="approve", input_fingerprint=request.input_fingerprint,
    ))


@router.post("/approve-all")
async def approve_all(request: DeliveryRequest):
    """One project approval, preserving resumable per-file save checkpoints."""
    async with workflow.workflow_locks[request.thread_id]:
        config, state = _state(request.thread_id)
        try:
            snapshot = critic.ensure_current_review(state.values, request.input_fingerprint)
            if critic.pending_repair_matches(snapshot, state.values.get("critic_pending_repair") or {}):
                raise critic.CriticError("Complete the pending repair review before approving files.")
            import ast
            for draft in state.values.get("drafts") or []:
                if not draft.get("syntax_valid"):
                    raise critic.CriticError("Generate valid Python drafts before approval.")
                ast.parse(draft["code"])
        except (critic.CriticError, OSError, ValueError, SyntaxError) as exc:
            raise HTTPException(409, "The complete candidate must have a current critic review and valid syntax before approval.") from exc
        for draft in state.values["drafts"]:
            if draft.get("status") == "COMMITTED":
                continue
            latest = workflow.agent_app.get_state(config)
            await workflow._submit_human_feedback(HumanFeedbackRequest(
                thread_id=request.thread_id, target_module=draft["target_module"], user_feedback="approve",
                input_fingerprint=latest.values["critic_report"]["input_fingerprint"]))
            latest = workflow.agent_app.get_state(config)
            saved = next(d for d in latest.values["drafts"] if d["target_module"] == draft["target_module"])
            if saved.get("status") != "COMMITTED":
                raise HTTPException(409, "A file could not be saved. Completed saves are preserved; refresh and retry approval.")
        return await workflow.get_agent_state(request.thread_id)
