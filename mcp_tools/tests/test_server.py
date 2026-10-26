"""Real MCP protocol sessions; inference and GitHub writes are never called."""
import asyncio
import json
from pathlib import Path

import httpx
import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from mcp_tools.api_client import BackendClient
from mcp_tools.server import create_server


TOOLS = {"extract_requirements", "generate_code", "get_run_state", "revise_code",
         "retry_missing_modules", "run_critic", "repair_findings", "approve_code",
         "generate_documentation", "create_draft_pr"}
REVIEW = {"thread_id": "test-run", "input_fingerprint": "a" * 64}
PUBLISH = {**REVIEW, "document_id": "b" * 64, "title": "Reviewed changes"}


def call(tmp_path, name, arguments, handler):
    async def perform():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            server = create_server(BackendClient(documents_root=tmp_path, client=http))
            async with create_connected_server_and_client_session(server) as session:
                return await session.call_tool(name, arguments)
    return asyncio.run(perform())


def test_discovery_exposes_typed_tools_and_side_effect_hints(tmp_path):
    async def perform():
        server = create_server(BackendClient(documents_root=tmp_path))
        async with create_connected_server_and_client_session(server) as session:
            result = await session.list_tools()
            assert {t.name for t in result.tools} == TOOLS
            assert all(t.inputSchema and t.outputSchema for t in result.tools)
            tools = {t.name: t for t in result.tools}
            assert tools["get_run_state"].annotations.readOnlyHint
            assert tools["create_draft_pr"].annotations.destructiveHint
    asyncio.run(perform())


@pytest.mark.parametrize("name,arguments,method,path,payload", [
    ("generate_code", {"request": {"thread_id": "test-run", "task_spec": "Implement requirements",
       "repository": {"url": "https://github.com/owner/repo.git"}}}, "POST", "/api/phase2/start",
       {"thread_id": "test-run", "task_spec": "Implement requirements", "repository": {
           "url": "https://github.com/owner/repo.git", "file_mappings": {}}}),
    ("get_run_state", {"thread_id": "test-run"}, "GET", "/api/phase2/state/test-run", None),
    ("revise_code", {"request": {"thread_id": "test-run", "user_feedback": "Validate input", "target_module": "accounts"}},
       "POST", "/api/phase2/feedback", {"thread_id": "test-run", "user_feedback": "Validate input", "target_module": "accounts"}),
    ("retry_missing_modules", {"thread_id": "test-run"}, "POST", "/api/phase2/feedback",
       {"thread_id": "test-run", "user_feedback": "Retry missing modules"}),
    ("run_critic", {"request": {"thread_id": "test-run"}}, "POST", "/api/critic/review", {"thread_id": "test-run"}),
    ("repair_findings", {"request": REVIEW}, "POST", "/api/critic/repair", REVIEW),
    ("approve_code", {"request": REVIEW, "confirm": True}, "POST", "/api/critic/approve-all", REVIEW),
    ("approve_code", {"request": REVIEW, "target_module": "accounts", "confirm": True}, "POST", "/api/critic/approve",
       {**REVIEW, "target_module": "accounts"}),
    ("generate_documentation", {"request": REVIEW}, "POST", "/api/delivery/documentation", REVIEW),
    ("create_draft_pr", {"request": PUBLISH, "confirm": True}, "POST", "/api/delivery/pull-request", PUBLISH),
])
def test_calls_forward_once_to_existing_endpoint(tmp_path, name, arguments, method, path, payload):
    seen = []
    def handler(request):
        seen.append(request)
        assert request.method == method and request.url.path == path
        if payload is not None:
            assert json.loads(request.content) == payload
        return httpx.Response(200, json={"status": "saved", "thread_id": "test-run"})
    result = call(tmp_path, name, arguments, handler)
    assert not result.isError, result.content
    assert result.structuredContent["data"]["status"] == "saved"
    assert len(seen) == 1


@pytest.mark.parametrize("name,arguments", [
    ("approve_code", {"request": REVIEW}),
    ("create_draft_pr", {"request": PUBLISH}),
    ("revise_code", {"request": {"thread_id": "test-run", "target_module": "accounts", "user_feedback": "approve"}}),
    ("get_run_state", {"thread_id": "../logs"}),
])
def test_rejected_actions_never_reach_backend(tmp_path, name, arguments):
    def handler(request):
        pytest.fail("Rejected action reached backend")
    assert call(tmp_path, name, arguments, handler).isError


def test_pdf_upload_uses_existing_multipart_route(tmp_path):
    (tmp_path / "requirements.pdf").write_bytes(b"%PDF-1.7 synthetic fixture")
    def handler(request):
        assert request.url.path == "/api/phase1/extract"
        assert b'name="files"' in request.content
        assert b"requirements.pdf" in request.content
        assert b"%PDF-1.7 synthetic fixture" in request.content
        return httpx.Response(200, json={"specs": {"features": []}})
    result = call(tmp_path, "extract_requirements", {"pdf_paths": ["requirements.pdf"]}, handler)
    assert not result.isError


@pytest.mark.parametrize("path", ["../outside.pdf", ".env", "missing.pdf"])
def test_upload_cannot_read_arbitrary_files(tmp_path, path):
    (tmp_path / ".env").write_text("PRIVATE_TOKEN=secret")
    def handler(request):
        pytest.fail("Invalid upload reached backend")
    assert call(tmp_path, "extract_requirements", {"pdf_paths": [path]}, handler).isError


@pytest.mark.parametrize("status", [409, 422, 429, 503])
def test_backend_failure_is_safe_and_never_retries(tmp_path, status):
    seen = []
    def handler(request):
        seen.append(request)
        return httpx.Response(status, json={"detail": "PRIVATE_TOKEN_AND_SOURCE"}, headers={"Retry-After": "30"})
    result = call(tmp_path, "run_critic", {"request": {"thread_id": "test-run"}}, handler)
    assert result.isError and len(seen) == 1
    text = " ".join(c.text for c in result.content)
    assert "PRIVATE_TOKEN_AND_SOURCE" not in text
    assert str(status) in text and '"retry_after": 30' in text


def test_timeout_preserves_uncertain_execution_state_without_retry(tmp_path):
    seen = []
    def handler(request):
        seen.append(request)
        raise httpx.ReadTimeout("PRIVATE_DIAGNOSTIC", request=request)
    result = call(tmp_path, "run_critic", {"request": {"thread_id": "test-run"}}, handler)
    assert result.isError and len(seen) == 1
    assert '"execution_state_unknown": true' in result.content[0].text
    assert "PRIVATE_DIAGNOSTIC" not in result.content[0].text


def test_remote_backend_and_embedded_credentials_are_rejected(tmp_path):
    for url in ("https://external.example", "http://secret@127.0.0.1:8000", "http://localhost:8000/api"):
        with pytest.raises(ValueError):
            BackendClient(base_url=url, documents_root=tmp_path)


def test_stdio_client_can_discover_tools_from_another_working_directory(tmp_path):
    import sys
    from datetime import timedelta
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    script = Path(__file__).resolve().parents[1] / "server.py"
    async def perform():
        parameters = StdioServerParameters(command=sys.executable, args=[str(script)], cwd=str(tmp_path))
        async with stdio_client(parameters) as (read, write):
            async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=30)) as session:
                await session.initialize()
                assert {tool.name for tool in (await session.list_tools()).tools} == TOOLS
    asyncio.run(perform())


def test_mcp_and_existing_api_share_the_same_checkpointed_workflow(tmp_path, monkeypatch):
    import importlib
    from unittest.mock import Mock
    from fastapi.testclient import TestClient
    monkeypatch.chdir(tmp_path)
    from backend import main
    from backend.agent import graph
    from backend.stages import critic
    phase1 = importlib.import_module("routers.phase1_router")
    monkeypatch.setattr(main, "CHECKPOINT_DB", tmp_path / "checkpoints.sqlite")
    specification = {"project_title": "MCP fixture", "cross_cutting_constraints": [], "features": [
        {"feature_name": name, "target_module": name, "business_rules": ["Return a stored value"],
         "functions": [], "exceptions": []} for name in ("accounts", "tasks")]}
    monkeypatch.setattr(phase1, "get_embedder", lambda: None)
    monkeypatch.setattr(phase1, "get_chroma_client", lambda: None)
    monkeypatch.setattr(phase1, "fallback_keys_available", lambda: True)
    extract = Mock(return_value=specification)
    monkeypatch.setattr(phase1, "run_extraction_pipeline", extract)
    coder = Mock()
    coder.generate_drafts_from_specs.return_value = [
        {"draft_id": name, "target_module": name, "code": "VALUE = 1\n", "syntax_valid": True,
         "validation_msg": "Syntax valid"} for name in ("accounts", "tasks")]
    monkeypatch.setattr(graph, "_build_coder", lambda: coder)
    def review(batch):
        return {"summary": "Offline fixture review", "coverage": [
            {"requirement_id": r["id"], "status": "met", "evidence": "VALUE = 1",
             "file_path": next(iter(batch["sources"]))} for r in batch["requirements"]],
             "findings": [], "limitations": []}
    monkeypatch.setattr(critic, "_review_model", review)
    (tmp_path / "input.pdf").write_bytes(b"%PDF-1.7 fixture")
    async def perform(ui):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app)) as http:
            server = create_server(BackendClient(documents_root=tmp_path, client=http))
            async with create_connected_server_and_client_session(server) as session:
                extracted = await session.call_tool("extract_requirements", {"pdf_paths": ["input.pdf"]})
                assert not extracted.isError, extracted.content
                specs = extracted.structuredContent["data"]["specs"]
                draft = await session.call_tool("generate_code", {"request": {
                    "thread_id": "mcp-flow", "task_spec": json.dumps(specs)}})
                assert not draft.isError, draft.content
                assert len(ui.get("/api/phase2/state/mcp-flow").json()["modules"]) == 2
                assert not list((tmp_path / "generated_src").glob("*.py"))
                report = await session.call_tool("run_critic", {"request": {"thread_id": "mcp-flow"}})
                assert not report.isError, report.content
                fingerprint = report.structuredContent["data"]["critic_report"]["input_fingerprint"]
                accepted = await session.call_tool("approve_code", {"request": {
                    "thread_id": "mcp-flow", "input_fingerprint": fingerprint}, "confirm": True})
                assert not accepted.isError, accepted.content
                assert accepted.structuredContent["data"]["status"] == "approved"
                assert (tmp_path / "generated_src/accounts.py").read_text() == "VALUE = 1\n"
                fingerprint = accepted.structuredContent["data"]["critic_report"]["input_fingerprint"]
                document = await session.call_tool("generate_documentation", {"request": {
                    "thread_id": "mcp-flow", "input_fingerprint": fingerprint}})
                assert not document.isError, document.content
                return document.structuredContent["data"]["documentation"]
    with TestClient(main.app) as ui:
        document = asyncio.run(perform(ui))
        assert ui.get("/api/phase2/state/mcp-flow").json()["documentation"] == document
    with TestClient(main.app) as ui:
        recovered = ui.get("/api/phase2/state/mcp-flow").json()
        assert recovered["documentation"] == document
        assert not recovered["documentation_stale"]
    assert extract.call_count == 1 and coder.generate_drafts_from_specs.call_count == 1
