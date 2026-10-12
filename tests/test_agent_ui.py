from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_agent_studio_displays_individual_module_files():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "frontend" / "app.py"))
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
