"""New-run cleanup is shared, bounded to generated_src, and fails visibly."""
import asyncio
from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi import HTTPException

from backend import workspace
from backend.models.schemas import ProjectSpecs, StartExecutionRequest
from backend.routers import phase2_router as api


@pytest.fixture
def output_dir(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GROQ_API_KEY", "offline")
    monkeypatch.setenv("GEMINI_API_KEY", "offline")
    output = tmp_path / "generated_src"
    output.mkdir()
    (output / "previous_run.py").write_text("OLD = True")
    return output


def specs():
    return ProjectSpecs(project_title="Test", features=[], cross_cutting_constraints=[])


def test_direct_coder_clears_output_before_generation(output_dir, monkeypatch):
    coder = Mock()

    def generate(_):
        assert output_dir.is_dir()
        assert list(output_dir.iterdir()) == []
        return []

    coder.generate_drafts_from_specs.side_effect = generate
    monkeypatch.setattr(api, "CoderAgent", Mock(return_value=coder))
    assert asyncio.run(api.generate_drafts(specs()))["status"] == "success"
    coder.generate_drafts_from_specs.assert_called_once()


@pytest.mark.parametrize("langgraph", [False, True])
def test_cleanup_failure_blocks_generation(output_dir, monkeypatch, langgraph):
    monkeypatch.setattr(workspace.shutil, "rmtree", Mock(side_effect=PermissionError("Locked file")))
    coder_factory = Mock()
    monkeypatch.setattr(api, "CoderAgent", coder_factory)
    graph = Mock()
    graph.get_state.return_value.values = {}
    monkeypatch.setattr(api, "agent_app", graph)
    with pytest.raises(HTTPException) as exc:
        if langgraph:
            asyncio.run(api.start_agent_workflow(StartExecutionRequest(thread_id="new", task_spec="Generate a module")))
        else:
            asyncio.run(api.generate_drafts(specs()))
    assert exc.value.status_code == 500
    assert "generation has not started" in exc.value.detail
    coder_factory.assert_not_called()
    graph.astream.assert_not_called()
    assert (output_dir / "previous_run.py").exists()


def test_rejects_output_resolving_outside_workspace(output_dir, monkeypatch, tmp_path):
    original_resolve = Path.resolve
    outside = tmp_path / "other_directory"
    outside.mkdir()
    (outside / "keep.py").write_text("KEEP = True")
    monkeypatch.setattr(Path, "resolve", lambda path, *a, **k:
        outside if path == output_dir else original_resolve(path, *a, **k))
    with pytest.raises(ValueError, match="inside the workspace"):
        workspace.clear_generated_workspace()
    assert (outside / "keep.py").exists()
    assert (output_dir / "previous_run.py").exists()


def test_missing_credentials_preserve_previous_output(output_dir, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(HTTPException):
        asyncio.run(api.generate_drafts(specs()))
    assert (output_dir / "previous_run.py").exists()
