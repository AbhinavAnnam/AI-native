"""Offline regressions for module output and the human review boundary."""
import asyncio
import json
from unittest.mock import Mock

import pytest
from fastapi import HTTPException

from backend.agent import graph
from backend.models.schemas import HumanFeedbackRequest, StartExecutionRequest
from backend.routers import phase2_router as api
from mcp_tools.file_tools import MCPToolServer

WRITE_MODULE = MCPToolServer.write_module_to_disk


SPEC = {"features": [
    {"feature_name": name, "target_module": name, "business_rules": [],
     "functions": [], "exceptions": []}
    for name in ("accounts", "work.tasks")
]}


@pytest.fixture
def workflow(monkeypatch, tmp_path):
    # New runs reset generated_src; never run that reset against real user files.
    monkeypatch.chdir(tmp_path)
    coder = Mock()
    coder.generate_drafts_from_specs.side_effect = lambda specs, **kwargs: [
        {"draft_id": f"draft_{f['target_module']}", "target_module": f["target_module"],
         "code": "def run():\n    return 1\n", "syntax_valid": True,
         "validation_msg": "ok", "tests_passed": False}
        for f in specs["features"]
    ]
    monkeypatch.setattr(graph, "_build_coder", lambda: coder)
    classifier = Mock(side_effect=AssertionError("Unnecessary classifier call"))
    monkeypatch.setattr(graph.router, "route", classifier)
    fallback = Mock(side_effect=AssertionError("Unexpected fallback"))
    monkeypatch.setattr(graph.router, "get_model", fallback)
    monkeypatch.setattr(api, "agent_app", graph.build_agent_graph())
    writer = Mock(return_value="generated_src/module.py")
    monkeypatch.setattr(MCPToolServer, "write_module_to_disk", writer)
    return coder, classifier, writer


def start(spec=SPEC):
    return asyncio.run(api.start_agent_workflow(StartExecutionRequest(
        thread_id="test", task_spec=json.dumps(spec))))


def feedback(text, modified_spec=None, target_module=None):
    return asyncio.run(api.submit_human_feedback(HumanFeedbackRequest(
        thread_id="test", user_feedback=text, modified_spec=modified_spec,
        target_module=target_module)))


def test_start_returns_separate_code_files_and_pauses(workflow):
    coder, classifier, writer = workflow
    result = start()
    assert result["status"] == "awaiting_approval"
    assert result["next_step"] == ("process_feedback",)
    assert [m["target_module"] for m in result["modules"]] == ["accounts", "work.tasks"]
    assert all(m["code"].startswith("def run") for m in result["modules"])
    assert [m["file_path"] for m in result["modules"]] == ["accounts.py", "work/tasks.py"]
    assert coder.generate_drafts_from_specs.call_count == 1
    classifier.assert_not_called()
    writer.assert_not_called()


def test_empty_feedback_does_not_regenerate(workflow):
    start()
    result = feedback("   ")
    assert result["status"] == "awaiting_approval"
    assert not result["is_completed"]
    assert workflow[0].generate_drafts_from_specs.call_count == 1


def test_revision_runs_once_then_approval_writes_each_module(workflow):
    start()
    result = feedback("Use UTC timestamps")
    assert result["status"] == "awaiting_approval"
    assert not result["is_completed"]
    assert workflow[0].generate_drafts_from_specs.call_count == 2
    assert workflow[0].generate_drafts_from_specs.call_args.kwargs["revision_notes"] == "Use UTC timestamps"
    first = feedback("approve", target_module="accounts")
    assert not first["is_completed"]
    assert first["modules"][0]["status"] == "COMMITTED"
    assert first["modules"][1]["status"] == "AWAITING_HUMAN_REVIEW"
    result = feedback("approve", target_module="work.tasks")
    assert result["is_completed"]
    assert result["status"] == "approved"
    assert [call.args[0] for call in workflow[2].call_args_list] == ["accounts", "work.tasks"]
    assert workflow[0].generate_drafts_from_specs.call_count == 2


def test_modified_spec_is_used_for_revision(workflow):
    start()
    changed = {"features": [SPEC["features"][1]]}
    result = feedback("Only generate tasks", changed)
    assert [m["target_module"] for m in result["modules"]] == ["work.tasks"]


def test_restarting_same_thread_is_rejected(workflow):
    start()
    with pytest.raises(HTTPException) as exc:
        start({"features": [SPEC["features"][0]]})
    assert exc.value.status_code == 409
    assert workflow[0].generate_drafts_from_specs.call_count == 1


def test_feedback_requires_existing_paused_thread(workflow):
    with pytest.raises(HTTPException) as exc:
        feedback("approve")
    assert exc.value.status_code == 404


def test_fallback_preserves_each_module(workflow, monkeypatch):
    workflow[0].generate_drafts_from_specs.return_value = []
    workflow[0].generate_drafts_from_specs.side_effect = None
    model = Mock()
    model.invoke.return_value.content = "def run():\n    return 2\n"
    monkeypatch.setattr(graph.router, "get_model", Mock(return_value=model))
    result = start()
    assert [m["target_module"] for m in result["modules"]] == ["accounts", "work.tasks"]
    assert model.invoke.call_count == 2
    assert all(m["code"].startswith("def run") for m in result["modules"])


def test_partial_generation_cannot_be_approved(workflow, monkeypatch):
    generate = workflow[0].generate_drafts_from_specs.side_effect
    workflow[0].generate_drafts_from_specs.side_effect = lambda specs, **kwargs: generate(specs)[:1]
    start()
    with pytest.raises(HTTPException) as exc:
        feedback("approve")
    assert exc.value.status_code == 400
    workflow[2].assert_not_called()


def test_approval_cannot_be_submitted_twice(workflow):
    start()
    feedback("approve", target_module="accounts")
    with pytest.raises(HTTPException) as exc:
        feedback("approve", target_module="accounts")
    assert exc.value.status_code == 409
    assert workflow[2].call_count == 1


def test_commit_failure_is_not_reported_as_success(workflow):
    start()
    workflow[2].side_effect = OSError("Write failed")
    result = feedback("approve", target_module="accounts")
    assert result["status"] == "commit_failed"
    assert not result["is_completed"]
    assert result["modules"][0]["status"] == "COMMIT_FAILED"
    assert "Write failed" in result["modules"][0]["commit_error"]
    workflow[2].side_effect = None
    result = feedback("approve", target_module="accounts")
    assert result["modules"][0]["status"] == "COMMITTED"
    assert not result["is_completed"]


def test_spec_only_revision_runs_once(workflow):
    start()
    result = feedback("  ", {"features": [SPEC["features"][0]]})
    assert len(result["modules"]) == 1
    assert not result["is_completed"]
    assert workflow[0].generate_drafts_from_specs.call_count == 2


def test_file_feedback_updates_only_selected_draft_before_approval(workflow):
    initial = start()
    changed = {**initial["modules"][0], "code": "def run():\n    return 42\n"}
    workflow[0].generate_single_module.return_value = changed
    result = feedback("Return 42", target_module="accounts")
    assert result["modules"][0]["code"] == changed["code"]
    assert result["modules"][1] == initial["modules"][1]
    assert result["status"] == "awaiting_approval"
    assert not result["is_completed"]
    workflow[0].generate_single_module.assert_called_once_with(
        SPEC["features"][0], revision_notes="Return 42",
        current_code=initial["modules"][0]["code"],
    )
    assert workflow[0].generate_drafts_from_specs.call_count == 1
    workflow[2].assert_not_called()
    feedback("approve", target_module="accounts")
    assert workflow[2].call_args_list[0].args[1] == changed["code"]


def test_followup_file_feedback_receives_latest_draft(workflow):
    start()
    draft = {"target_module": "accounts", "draft_id": "draft_accounts",
             "code": "VALUE = 42", "syntax_valid": True}
    workflow[0].generate_single_module.return_value = draft
    feedback("Return 42", target_module="accounts")
    feedback("Also add logging", target_module="accounts")
    assert workflow[0].generate_single_module.call_args.kwargs["current_code"] == "VALUE = 42"


@pytest.mark.parametrize("target, notes", [("missing", "Change this"), (None, "approve")])
def test_invalid_file_feedback_cannot_commit_or_regenerate(workflow, target, notes):
    start()
    with pytest.raises(HTTPException) as exc:
        feedback(notes, target_module=target)
    assert exc.value.status_code == 400
    workflow[0].generate_single_module.assert_not_called()
    workflow[2].assert_not_called()


def test_approving_one_file_keeps_other_files_open_for_revision(workflow):
    initial = start()
    result = feedback("approve", target_module="accounts")
    assert result["status"] == "awaiting_approval"
    assert not result["is_completed"]
    assert [c.args[0] for c in workflow[2].call_args_list] == ["accounts"]
    changed = {**initial["modules"][1], "code": "VALUE = 42"}
    workflow[0].generate_single_module.return_value = changed
    revised = feedback("Use 42", target_module="work.tasks")
    assert revised["modules"][0]["status"] == "COMMITTED"
    assert revised["modules"][0]["code"] == initial["modules"][0]["code"]
    assert revised["modules"][1]["status"] == "AWAITING_HUMAN_REVIEW"
    completed = feedback("approve", target_module="work.tasks")
    assert completed["is_completed"]
    assert completed["status"] == "approved"
    assert workflow[2].call_args_list[-1].args == ("work.tasks", "VALUE = 42")
    assert workflow[0].generate_drafts_from_specs.call_count == 1


def test_revising_saved_file_requires_approval_again(workflow):
    initial = start()
    feedback("approve", target_module="accounts")
    workflow[0].generate_single_module.return_value = {
        **initial["modules"][0], "code": "VALUE = 42", "status": "COMMITTED",
    }
    result = feedback("Use 42", target_module="accounts")
    assert result["modules"][0]["status"] == "AWAITING_HUMAN_REVIEW"
    assert workflow[2].call_count == 1
    feedback("approve", target_module="accounts")
    assert workflow[2].call_args_list[-1].args == ("accounts", "VALUE = 42")


def test_valid_file_can_be_approved_while_another_draft_is_missing(workflow):
    generate = workflow[0].generate_drafts_from_specs.side_effect
    workflow[0].generate_drafts_from_specs.side_effect = lambda specs, **kwargs: generate(specs)[:1]
    start()
    result = feedback("approve", target_module="accounts")
    assert result["modules"][0]["status"] == "COMMITTED"
    assert not result["is_completed"]
    with pytest.raises(HTTPException) as exc:
        feedback("approve", target_module="work.tasks")
    assert exc.value.status_code == 400


def test_invalid_file_cannot_be_approved(workflow):
    generate = workflow[0].generate_drafts_from_specs.side_effect
    workflow[0].generate_drafts_from_specs.side_effect = lambda specs, **kwargs: [
        {**d, "syntax_valid": False} for d in generate(specs)
    ]
    start()
    with pytest.raises(HTTPException) as exc:
        feedback("approve", target_module="accounts")
    assert exc.value.status_code == 400
    workflow[2].assert_not_called()


def test_refresh_restores_separate_drafts_without_running_agent(workflow):
    initial = start()
    result = asyncio.run(api.get_agent_state("test"))
    assert result["modules"] == initial["modules"]
    assert not result["is_completed"]
    assert workflow[0].generate_drafts_from_specs.call_count == 1
    workflow[2].assert_not_called()


def test_file_revision_fallback_does_not_regenerate_other_modules(workflow, monkeypatch):
    initial = start()
    workflow[0].generate_single_module.side_effect = RuntimeError("Coder unavailable")
    model = Mock()
    model.invoke.return_value.content = "VALUE = 42"
    monkeypatch.setattr(graph.router, "get_model", Mock(return_value=model))
    result = feedback("Use 42", target_module="accounts")
    assert result["modules"][0]["code"] == "VALUE = 42"
    assert result["modules"][1] == initial["modules"][1]
    model.invoke.assert_called_once()
    prompt = model.invoke.call_args.args[0][0].content
    assert initial["modules"][0]["code"] in prompt
    assert "Use 42" in prompt
    workflow[2].assert_not_called()


def test_ui_start_and_file_revision_render_before_any_approval(workflow, monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace
    from streamlit.testing.v1 import AppTest
    import requests

    def post(url, json, **kwargs):
        if url.endswith("/start"):
            result = asyncio.run(api.start_agent_workflow(StartExecutionRequest(**json)))
        else:
            result = asyncio.run(api.submit_human_feedback(HumanFeedbackRequest(**json)))
        return SimpleNamespace(status_code=200, json=lambda: result)

    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(requests, "get", lambda *a, **k: SimpleNamespace(
        status_code=200, json=lambda: {"logs": "Offline UI test"}))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "frontend" / "app.py"))
    app.session_state["agent_thread_id"] = "test"
    app.session_state["extracted_specs"] = SPEC
    app.run()
    next(b for b in app.button if "Start Agent" in b.label).click().run()
    assert not app.exception
    assert len([c for c in app.code if c.language == "python"]) == 2
    assert len(app.get("download_button")) == 2
    workflow[2].assert_not_called()

    workflow[0].generate_single_module.return_value = {
        "target_module": "accounts", "draft_id": "draft_accounts",
        "code": "VALUE = 42", "syntax_valid": True,
    }
    app.text_area(key="file_feedback_test_accounts").set_value("Use 42").run()
    app.button(key="revise_test_accounts").click().run()
    assert not app.exception
    assert [c.value for c in app.code if c.language == "python"] == [
        "VALUE = 42", "def run():\n    return 1",
    ]
    assert app.session_state["agent_status"] == "awaiting_approval"
    workflow[2].assert_not_called()

    app.button(key="approve_test_accounts").click().run()
    assert not app.exception
    assert workflow[2].call_args.args == ("accounts", "VALUE = 42")
    assert workflow[2].call_count == 1
    assert app.button(key="approve_test_accounts").disabled
    assert not app.button(key="approve_test_work.tasks").disabled
    assert not app.session_state["agent_completed"]

    app.button(key="approve_test_work.tasks").click().run()
    assert not app.exception
    assert [c.args[0] for c in workflow[2].call_args_list] == ["accounts", "work.tasks"]
    assert app.session_state["agent_completed"]
    assert app.session_state["agent_status"] == "approved"
    app.run()
    assert not app.exception
    assert workflow[2].call_count == 2
    assert not [b for b in app.button if b.key and b.key.startswith("approve_")]


def test_file_approval_writes_only_selected_file_on_disk(workflow, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(MCPToolServer, "write_module_to_disk", WRITE_MODULE)
    start()
    feedback("approve", target_module="accounts")
    assert (tmp_path / "generated_src" / "accounts.py").read_text() == "def run():\n    return 1\n"
    assert not (tmp_path / "generated_src" / "work" / "tasks.py").exists()
    feedback("approve", target_module="work.tasks")
    assert (tmp_path / "generated_src" / "work" / "tasks.py").exists()


def test_new_run_clears_old_files_but_reviews_preserve_current_files(workflow, monkeypatch, tmp_path):
    output = tmp_path / "generated_src"
    (output / "old_package").mkdir(parents=True)
    (output / "old_package" / "old.py").write_text("OLD = True")
    unrelated = tmp_path / "keep.py"
    unrelated.write_text("KEEP = True")
    monkeypatch.setattr(MCPToolServer, "write_module_to_disk", WRITE_MODULE)
    initial = start()
    assert output.is_dir()
    assert list(output.iterdir()) == []
    assert unrelated.read_text() == "KEEP = True"

    feedback("approve", target_module="accounts")
    saved = output / "accounts.py"
    assert saved.exists()
    workflow[0].generate_single_module.return_value = {**initial["modules"][1], "code": "VALUE = 42"}
    feedback("Use 42", target_module="work.tasks")
    asyncio.run(api.get_agent_state("test"))
    assert saved.exists()
    with pytest.raises(HTTPException):
        start()
    assert saved.exists()

    asyncio.run(api.start_agent_workflow(StartExecutionRequest(
        thread_id="next-run", task_spec=json.dumps(SPEC))))
    assert list(output.iterdir()) == []
