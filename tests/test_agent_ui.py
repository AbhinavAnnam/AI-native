from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_delivery_preview_does_not_publish_until_button_clicked(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    import requests
    doc = {"document_id": "a" * 64, "input_fingerprint": "b" * 64,
           "requirements_assessed": 3, "findings": 1, "markdown": "# Report\nKnown issue. Tests not run.",
           "filename": "AI_NATIVE_REVIEW_demo.md"}
    post = Mock(return_value=SimpleNamespace(status_code=200, json=lambda: {
        "documentation": doc, "documentation_stale": False, "pull_request": None}))
    monkeypatch.setattr(requests, "post", post)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "frontend/app.py"), default_timeout=10)
    app.session_state["agent_status"] = "approved"
    app.session_state["agent_repository"] = {"work_branch": "ai-native/demo", "base_branch": "main", "baseline_commit": "f" * 40}
    app.session_state["critic_report"] = {"input_fingerprint": "b" * 64, "verdict": "needs_review", "summary": "Review",
        "coverage": [], "findings": [], "limitations": []}
    app.run()
    assert not app.exception
    assert not post.called
    app.button(key="delivery_generate").click().run()
    assert not app.exception
    assert post.call_count == 1
    assert post.call_args.args[0].endswith("/delivery/documentation")
    assert not app.button(key="delivery_publish").disabled
    assert any("Automated tests were not run" in warning.value for warning in app.warning)
    post.return_value = SimpleNamespace(status_code=200, json=lambda: {
        "documentation": doc, "documentation_stale": False, "pull_request": {
            "number": 3, "url": "https://github.com/owner/repo/pull/3"}})
    app.button(key="delivery_publish").click().run()
    assert not app.exception
    assert post.call_count == 2
    assert post.call_args.args[0].endswith("/delivery/pull-request")
    assert post.call_args.kwargs["json"]["document_id"] == doc["document_id"]
    assert app.button(key="delivery_publish").disabled


def test_retry_missing_button_submits_one_action_and_keeps_successful_file(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    import requests
    saved = {"target_module": "accounts", "code": "VALUE = 1", "syntax_valid": True}
    recovered = {"target_module": "tasks", "code": "TASK = 2", "syntax_valid": True}
    post = Mock(return_value=SimpleNamespace(status_code=200, json=lambda: {
        "status": "awaiting_approval", "modules": [saved, recovered], "missing_modules": []}))
    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: SimpleNamespace(status_code=200, json=lambda: {"logs": "Offline"}))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "frontend/app.py"), default_timeout=10)
    app.session_state["agent_status"] = "awaiting_approval"
    app.session_state["agent_modules"] = [saved]
    app.session_state["agent_missing_modules"] = ["tasks"]
    app.run()
    app.button(key="agent_retry_missing").click().run()
    assert not app.exception
    assert post.call_count == 1
    assert post.call_args.args[0].endswith("/phase2/feedback")
    assert post.call_args.kwargs["json"]["user_feedback"] == "Retry missing modules"
    assert app.session_state["agent_modules"][0] == saved
    assert app.session_state["agent_missing_modules"] == []
    assert not [button for button in app.button if button.key == "agent_retry_missing"]


def test_agent_studio_displays_individual_module_files():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "frontend" / "app.py"), default_timeout=10)
    app.session_state["agent_output"] = ""
    app.session_state["agent_status"] = "awaiting_approval"
    app.session_state["agent_modules"] = [
        {"target_module": "accounts", "file_path": "accounts.py", "code": "ACCOUNT = 1",
         "syntax_valid": True, "validation_msg": "ok"},
        {"target_module": "work.tasks", "file_path": "work/tasks.py", "code": "TASK = 2",
         "syntax_valid": True, "validation_msg": "ok"},
    ]
    app.run()
    assert not app.exception
    assert [item.value for item in app.code if item.language == "python"] == ["ACCOUNT = 1", "TASK = 2"]
    assert len(app.get("download_button")) == 2
    assert any("work/tasks.py" in item.label for item in app.expander)
    assert next(b for b in app.button if "Start Agent" in b.label).disabled
    assert next(b for b in app.button if "Submit Feedback" in b.label).disabled


def test_legacy_combined_output_is_replaced_by_refreshed_file_cards(monkeypatch):
    from types import SimpleNamespace
    import requests

    modules = [{"target_module": "accounts", "code": "VALUE = 1", "syntax_valid": True}]
    monkeypatch.setattr(requests, "get", lambda *a, **k: SimpleNamespace(
        status_code=200, json=lambda: {"modules": modules, "status": "awaiting_approval", "is_completed": False}))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "frontend" / "app.py"))
    app.session_state["agent_output"] = "# MODULE: accounts\nVALUE = 1"
    app.session_state["agent_status"] = "awaiting_approval"
    app.run()
    assert not [c for c in app.code if c.language == "python"]
    next(b for b in app.button if b.label == "Refresh draft files").click().run()
    assert not app.exception
    assert [c.value for c in app.code if c.language == "python"] == ["VALUE = 1"]


def test_missing_backend_thread_preserves_downloads_and_stops_approval_retries(monkeypatch):
    from types import SimpleNamespace
    import requests

    monkeypatch.setattr(requests, "post", lambda *a, **k: SimpleNamespace(
        status_code=404, text='{"detail":"Agent thread not found."}'))
    monkeypatch.setattr(requests, "get", lambda *a, **k: SimpleNamespace(
        status_code=200, json=lambda: {"logs": "Offline test"}))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "frontend/app.py"))
    app.session_state["agent_thread_id"] = "lost-thread"
    app.session_state["agent_status"] = "awaiting_approval"
    app.session_state["agent_modules"] = [{
        "draft_id": "draft_accounts", "target_module": "accounts", "file_path": "accounts.py",
        "code": "VALUE = 42", "syntax_valid": True, "status": "AWAITING_HUMAN_REVIEW",
    }]
    app.run()
    app.button(key="critic_review").click().run()
    assert not app.exception
    assert app.session_state["agent_status"] == "review_missing"
    assert app.session_state["agent_modules"][0]["code"] == "VALUE = 42"
    assert len(app.get("download_button")) == 1
    assert not [button for button in app.button if button.key and button.key.startswith("approve_")]
    assert any("download" in error.value.lower() for error in app.error)


def test_critic_quota_pause_does_not_start_automatic_requests(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    import requests
    import time
    state = {"status": "awaiting_approval", "modules": [{"target_module": "accounts", "code": "VALUE = 1", "syntax_valid": True}]}
    post = Mock(side_effect=[
        SimpleNamespace(status_code=429, json=lambda: {"detail": {"retry_after": 1}}),
        SimpleNamespace(status_code=200, json=lambda: state),
    ])
    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: SimpleNamespace(status_code=200, json=lambda: state))
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "frontend/app.py"))
    app.session_state["agent_status"] = state["status"]
    app.session_state["agent_modules"] = state["modules"]
    app.run()
    assert not app.number_input
    assert app.button(key="critic_review").label == "Run Critic"
    app.button(key="critic_review").click().run()
    assert not app.exception
    assert post.call_count == 1
    assert app.warning
    assert "max_batches" not in post.call_args.kwargs["json"]
    assert app.session_state["active_phase"] == "Phase 3: Critic & Repair"


def test_failed_new_extraction_clears_old_specification_before_new_generation(monkeypatch):
    from types import SimpleNamespace
    import requests
    import streamlit as st

    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: [
        SimpleNamespace(name="synthetic.pdf", getvalue=lambda: b"offline-fixture")])
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: SimpleNamespace(status_code=500, text="Extraction incomplete"))
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: SimpleNamespace(
        status_code=200, json=lambda: {"logs": "Offline test"}))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "frontend/app.py"))
    app.session_state["extracted_specs"] = {"project_title": "Old document", "features": [{
        "target_module": "old_accounts", "business_rules": ["Old requirement"], "functions": []}],
        "cross_cutting_constraints": []}
    app.run()
    assert "Old document" in app.session_state["agent_task_spec"]
    next(button for button in app.button if "Process Documents" in button.label).click().run()
    assert not app.exception
    assert app.session_state["extracted_specs"] is None
    assert app.session_state["agent_task_spec"] == ""


def test_phase2_shows_which_extracted_features_share_a_module():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "frontend/app.py"), default_timeout=10)
    app.session_state["agent_status"] = "awaiting_approval"
    app.session_state["agent_module_groups"] = {"organizations": ["organizations", "organization_settings"]}
    app.run()
    assert not app.exception
    assert any("2 extracted features into 1 code modules" in caption.value for caption in app.caption)
    assert any(expander.label == "Merged feature groups" for expander in app.expander)


def test_same_repair_button_can_resume_pending_review_at_round_limit():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "frontend/app.py"), default_timeout=10)
    app.session_state["agent_status"] = "awaiting_approval"
    app.session_state["agent_modules"] = [{"target_module": "accounts", "code": "VALUE = 1", "syntax_valid": True}]
    app.session_state["critic_pending_repair"] = True
    app.session_state["critic_repair_in_progress"] = False
    app.session_state["critic_repair_rounds"] = 2
    app.session_state["critic_stale"] = True
    app.session_state["critic_report"] = {"input_fingerprint": "a" * 64, "verdict": "needs_review",
        "summary": "Pending repaired candidate", "coverage": [], "findings": [], "limitations": []}
    app.run()
    assert not app.exception
    assert app.button(key="critic_repair").label == "Repair findings"
    assert not app.button(key="critic_repair").disabled
