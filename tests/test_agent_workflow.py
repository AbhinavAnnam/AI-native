"""Offline regressions for module output and the human review boundary."""
import asyncio
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi import HTTPException

from backend.agent import graph
from backend.models.schemas import HumanFeedbackRequest, StartExecutionRequest
from backend.routers import phase2_router as api
from backend.routers import critic_router as critic_api
from backend.models.critic import ReviewRequest, ApprovalRequest
from backend.stages import critic
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
    monkeypatch.setattr(critic, "_review_model", lambda batch: {
        "summary": "Offline review", "coverage": [
            {"requirement_id": r["id"], "status": "met", "evidence": "Fixture code", "file_path": next(iter(batch["sources"]))}
            for r in batch["requirements"]], "findings": [], "limitations": [],
    })
    writer = Mock(return_value="generated_src/module.py")
    monkeypatch.setattr(MCPToolServer, "write_module_to_disk", writer)
    return coder, classifier, writer


def start(spec=SPEC):
    return asyncio.run(api.start_agent_workflow(StartExecutionRequest(
        thread_id="test", task_spec=json.dumps(spec))))


def feedback(text, modified_spec=None, target_module=None):
    if graph._is_approval(text):
        state = api.agent_app.get_state({"configurable": {"thread_id": "test"}})
        if state.values and state.next and {d["target_module"] for d in state.values.get("drafts", [])} == {
            f["target_module"] for f in state.values["base_specs"]["features"]
        }:
            asyncio.run(critic_api.review(ReviewRequest(thread_id="test")))
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


def test_retry_missing_modules_preserves_successes_and_does_not_regenerate_them(workflow):
    coder = workflow[0]
    original = coder.generate_drafts_from_specs.side_effect
    coder.failed_modules = {"work.tasks"}
    coder.generate_drafts_from_specs.side_effect = lambda specs, **kw: [
        d for d in original(specs, **kw) if d["target_module"] == "accounts"]
    initial = start()
    assert initial["missing_modules"] == ["work.tasks"]
    saved = initial["modules"][0]
    coder.failed_modules = set()
    coder.generate_drafts_from_specs.side_effect = original
    restored = feedback("Retry missing modules")
    assert [f["target_module"] for f in coder.generate_drafts_from_specs.call_args.args[0]["features"]] == ["work.tasks"]
    assert restored["missing_modules"] == []
    assert restored["modules"][0] == saved
    assert len(restored["modules"]) == 2
    calls = coder.generate_drafts_from_specs.call_count
    feedback("Retry missing modules")
    assert coder.generate_drafts_from_specs.call_count == calls
    workflow[2].assert_not_called()


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


def test_bounded_coder_failure_does_not_restart_legacy_fallback(workflow, monkeypatch):
    generate = workflow[0].generate_drafts_from_specs.side_effect
    workflow[0].generate_drafts_from_specs.side_effect = lambda specs, **kwargs: generate(specs)[:1]
    workflow[0].failed_modules = {"work.tasks"}
    fallback = Mock(side_effect=AssertionError("Bounded failure must not restart model loops"))
    monkeypatch.setattr(graph, "_fallback_generation", fallback)
    result = start()
    assert [m["target_module"] for m in result["modules"]] == ["accounts"]
    assert "Generation failed for: work.tasks" in result["generated_output"] or "work.tasks" in (
        api.agent_app.get_state({"configurable": {"thread_id": "test"}}).values["messages"][-1].content)
    assert not fallback.called


def test_failed_file_revision_keeps_previous_candidate(workflow, monkeypatch):
    initial = start()
    workflow[0].generate_single_module.return_value = {
        "target_module": "accounts", "code": "", "syntax_valid": False, "generation_failed": True}
    monkeypatch.setattr(graph, "_fallback_generation", Mock(side_effect=AssertionError("No retry chain")))
    result = feedback("Update account handling", target_module="accounts")
    assert [m["code"] for m in result["modules"]] == [m["code"] for m in initial["modules"]]


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


def test_incomplete_candidate_requires_complete_generation_before_review_and_save(workflow):
    generate = workflow[0].generate_drafts_from_specs.side_effect
    workflow[0].generate_drafts_from_specs.side_effect = lambda specs, **kwargs: generate(specs)[:1]
    start()
    with pytest.raises(HTTPException) as exc:
        feedback("approve", target_module="accounts")
    assert exc.value.status_code == 409
    workflow[2].assert_not_called()
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
        elif url.endswith("/review"):
            result = asyncio.run(critic_api.review(ReviewRequest(**json)))
        elif url.endswith("/approve"):
            result = asyncio.run(critic_api.approve(ApprovalRequest(**json)))
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

    assert not [b for b in app.button if b.key and b.key.startswith("approve_")]
    app.button(key="critic_review").click().run()
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
    assert all(b.disabled for b in app.button if b.key and b.key.startswith("approve_"))


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


@pytest.fixture
def repository_workflow(workflow, monkeypatch, tmp_path):
    from backend.models.repository import RepositoryWorkspace
    from backend.stages.repository import file_hash

    repo = tmp_path / "workspace" / "demo" / "repo"
    path = repo / "src/organization_demo/organizations.py"
    path.parent.mkdir(parents=True)
    path.write_text("VALUE = 1\n")
    workspace = RepositoryWorkspace(
        run_id="demo", url="https://github.com/AbhinavAnnam/organization-demo.git",
        repo_dir=str(repo), base_branch="main", baseline_commit="a" * 40,
        work_branch="ai-native/demo", file_mappings={"accounts": "src/organization_demo/organizations.py"},
        original_codes={"accounts": path.read_text()}, original_hashes={"accounts": file_hash(path)},
        context="Existing organization package and tests",
    )
    prepare = Mock(return_value=workspace)
    monkeypatch.setattr(graph, "prepare_repository", prepare)
    return workspace, prepare


def start_repository(file_mappings=None):
    return asyncio.run(api.start_agent_workflow(StartExecutionRequest(
        thread_id="test", task_spec=json.dumps({"features": [SPEC["features"][0]]}),
        repository={"url": "https://github.com/AbhinavAnnam/organization-demo.git",
                    "file_mappings": file_mappings or {"accounts": "src/organization_demo/organizations.py"}},
    )))


def test_repository_start_loads_existing_code_and_preserves_generated_output(repository_workflow, workflow, tmp_path):
    workspace, prepare = repository_workflow
    output = tmp_path / "generated_src"
    output.mkdir()
    (output / "keep.py").write_text("KEEP = True")
    result = start_repository()
    assert (output / "keep.py").exists()
    assert result["repository"]["baseline_commit"] == workspace.baseline_commit
    assert result["modules"][0]["file_path"] == "src/organization_demo/organizations.py"
    assert workflow[0].generate_drafts_from_specs.call_args.kwargs["current_codes"] == {"accounts": "VALUE = 1\n"}
    assert workflow[0].generate_drafts_from_specs.call_args.kwargs["repository_context"] == workspace.context
    assert prepare.call_count == 1
    assert (Path(workspace.repo_dir) / workspace.file_mappings["accounts"]).read_text() == "VALUE = 1\n"


def test_repository_approval_saves_to_clone_and_revision_does_not_reclone(repository_workflow, workflow):
    workspace, prepare = repository_workflow
    start_repository()
    workflow[0].generate_single_module.return_value = {
        "draft_id": "draft_accounts", "target_module": "accounts", "code": "VALUE = 42\n", "syntax_valid": True,
    }
    feedback("Return 42", target_module="accounts")
    result = feedback("approve", target_module="accounts")
    assert result["is_completed"]
    assert (Path(workspace.repo_dir) / workspace.file_mappings["accounts"]).read_text() == "VALUE = 42\n"
    assert prepare.call_count == 1
    workflow[2].assert_not_called()
    assert asyncio.run(api.get_agent_state("test"))["repository"]["work_branch"] == workspace.work_branch


def test_repository_mapping_must_match_spec_before_clone(repository_workflow, workflow):
    result = start_repository({"wrong_module": "src/organization_demo/organizations.py"})
    assert result["status"] == "repository_failed"
    assert result["next_step"] == ()
    repository_workflow[1].assert_not_called()
    workflow[0].generate_drafts_from_specs.assert_not_called()


def test_repository_clone_failure_stops_before_generation(repository_workflow, workflow):
    from backend.tools.git_provider import RepositoryError
    repository_workflow[1].side_effect = RepositoryError("Repository clone failed")
    result = start_repository()
    assert result["status"] == "repository_failed"
    assert "clone failed" in result["error"]
    workflow[0].generate_drafts_from_specs.assert_not_called()


def test_repository_approval_refuses_external_edits(repository_workflow):
    workspace, _ = repository_workflow
    start_repository()
    path = Path(workspace.repo_dir) / workspace.file_mappings["accounts"]
    path.write_text("MANUAL = True\n")
    result = feedback("approve", target_module="accounts")
    assert result["status"] == "commit_failed"
    assert "changed" in result["modules"][0]["commit_error"]
    assert path.read_text() == "MANUAL = True\n"


def test_repository_mode_ui_submits_mapping_and_saves_to_repository(repository_workflow, monkeypatch):
    from types import SimpleNamespace
    from streamlit.testing.v1 import AppTest
    import requests

    def post(url, json, **kwargs):
        if url.endswith("/start"):
            result = asyncio.run(api.start_agent_workflow(StartExecutionRequest(**json)))
        elif url.endswith("/review"):
            result = asyncio.run(critic_api.review(ReviewRequest(**json)))
        elif url.endswith("/approve"):
            result = asyncio.run(critic_api.approve(ApprovalRequest(**json)))
        else:
            result = asyncio.run(api.submit_human_feedback(HumanFeedbackRequest(**json)))
        return SimpleNamespace(status_code=200, json=lambda: result)

    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(requests, "get", lambda *a, **k: SimpleNamespace(
        status_code=200, json=lambda: {"logs": "Offline repository UI test"}))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "frontend/app.py"))
    app.session_state["agent_thread_id"] = "test"
    app.session_state["extracted_specs"] = {"features": [SPEC["features"][0]]}
    app.run()
    app.checkbox(key="agent_use_repository").check().run()
    app.text_input(key="agent_repository_url").set_value(repository_workflow[0].url)
    app.text_area(key="agent_file_mappings").set_value(json.dumps(repository_workflow[0].file_mappings))
    next(b for b in app.button if "Start Agent" in b.label).click().run()
    assert not app.exception
    assert any("src/organization_demo/organizations.py" in item.label for item in app.expander)
    assert app.session_state["agent_repository"]["work_branch"] == "ai-native/demo"
    app.button(key="critic_review").click().run()
    app.button(key="approve_test_accounts").click().run()
    assert not app.exception
    assert any("repository" in item.value.lower() and "saved" in item.value.lower() for item in app.success)
    assert not any("generated_src" in item.value for item in app.success)


def test_repository_modified_spec_cannot_add_unmapped_targets(repository_workflow):
    start_repository()
    with pytest.raises(HTTPException) as exc:
        feedback("Add another module", modified_spec=SPEC)
    assert exc.value.status_code == 400


def test_candidate_version_changes_with_code_and_survives_approval(repository_workflow, workflow):
    initial = start_repository()
    workflow[0].generate_single_module.return_value = {
        "draft_id": "draft_accounts", "target_module": "accounts", "code": "VALUE = 42\n", "syntax_valid": True,
    }
    revised = feedback("Use 42", target_module="accounts")
    assert revised["candidate_version"] != initial["candidate_version"]
    approved = feedback("approve", target_module="accounts")
    assert approved["candidate_version"] == revised["candidate_version"]
    assert asyncio.run(api.get_agent_state("test"))["candidate_version"] == revised["candidate_version"]


def test_repository_initial_fallback_receives_original_file(repository_workflow, workflow, monkeypatch):
    workflow[0].generate_drafts_from_specs.side_effect = RuntimeError("Primary coder unavailable")
    model = Mock()
    model.invoke.return_value.content = "VALUE = 2\n"
    monkeypatch.setattr(graph.router, "get_model", Mock(return_value=model))
    result = start_repository()
    prompt = model.invoke.call_args.args[0][0].content
    assert "VALUE = 1" in prompt
    assert "src/organization_demo/organizations.py" in prompt
    assert result["modules"][0]["code"] == "VALUE = 2"


def test_repository_start_without_manual_mapping_uses_specification(repository_workflow, workflow):
    result = asyncio.run(api.start_agent_workflow(StartExecutionRequest(
        thread_id="test", task_spec=json.dumps({"features": [SPEC["features"][0]]}),
        repository={"url": repository_workflow[0].url},
    )))
    assert result["status"] == "awaiting_approval"
    assert result["modules"][0]["file_path"] == "src/organization_demo/organizations.py"
    assert repository_workflow[1].call_args.kwargs["specification"]["features"] == [SPEC["features"][0]]


def test_repository_ui_accepts_url_without_manual_mapping(repository_workflow, monkeypatch):
    from types import SimpleNamespace
    from streamlit.testing.v1 import AppTest
    import requests

    submitted = []
    def post(url, json, **kwargs):
        submitted.append(json)
        result = asyncio.run(api.start_agent_workflow(StartExecutionRequest(**json)))
        return SimpleNamespace(status_code=200, json=lambda: result)

    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(requests, "get", lambda *a, **k: SimpleNamespace(
        status_code=200, json=lambda: {"logs": "Offline test"}))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "frontend/app.py"))
    app.session_state["agent_thread_id"] = "test"
    app.session_state["extracted_specs"] = {"features": [SPEC["features"][0]]}
    app.run()
    app.checkbox(key="agent_use_repository").check().run()
    app.text_input(key="agent_repository_url").set_value(repository_workflow[0].url)
    next(b for b in app.button if "Start Agent" in b.label).click().run()
    assert not app.exception
    assert len(submitted) == 1
    assert submitted[0]["repository"]["file_mappings"] == {}
    assert app.session_state["agent_status"] == "awaiting_approval"
