"""Guard against retry amplification and false truncation retries."""
from unittest.mock import Mock

import pytest

from backend.agent import router
from phase2_coder import coder
from phase2_coder.coder import CoderAgent, _looks_truncated


@pytest.mark.parametrize("code", [
    "# Customer's account (optional\ndef account():\n    return 1\n",
    'def account():\n    """The customer\'s account."""\n    return 1\n',
])
def test_valid_python_is_not_rejected_as_truncated(code):
    assert not _looks_truncated(code)


def test_incomplete_python_is_still_rejected():
    assert _looks_truncated("def account():\n    return [1,")


@pytest.mark.parametrize("classification", [False, True])
def test_transient_failures_have_bounded_backoff(monkeypatch, classification):
    factory = Mock()
    factory.return_value.invoke.side_effect = RuntimeError("503 unavailable")
    monkeypatch.setattr(router, "ChatGoogleGenerativeAI", factory)
    sleep = Mock()
    monkeypatch.setattr(router.time, "sleep", sleep)
    with pytest.raises(RuntimeError, match="503"):
        if classification:
            router.LLMRouter()._run_with_fallback(["one", "two"], 0, "task")
        else:
            router._RetryingChatModel(["one", "two"], "offline", 0).invoke([])
    assert factory.call_count == 4
    assert [c.args[0] for c in sleep.call_args_list] == [1.0, 1.0]
    assert all(c.kwargs["max_retries"] == 0 for c in factory.call_args_list)
    assert all(c.kwargs["timeout"] <= 60 for c in factory.call_args_list)


def test_coder_sdk_does_not_add_another_retry_layer(monkeypatch):
    monkeypatch.setattr(coder, "OpenAI", Mock())
    factory = Mock()
    monkeypatch.setattr(coder.genai, "Client", factory)
    CoderAgent("offline", "offline")
    options = factory.call_args.kwargs["http_options"]
    assert options.retry_options.attempts == 1
    assert options.timeout == 60_000


@pytest.mark.parametrize("message, expected_calls", [
    ("503 unavailable", 1), ("429 quota", 1),
    ("400 max_output_tokens exceeds the limit", 2),
])
def test_gemini_only_retries_smaller_budget_for_token_errors(message, expected_calls):
    agent = object.__new__(CoderAgent)
    agent.gemini_client = Mock()
    generate = agent.gemini_client.models.generate_content
    generate.side_effect = RuntimeError(message)
    with pytest.raises(RuntimeError):
        agent._gemini_generate_text("test-model", "task")
    assert generate.call_count == expected_calls


def test_revision_prompt_includes_current_code_and_feedback(monkeypatch, tmp_path):
    agent = object.__new__(CoderAgent)
    agent.staging_dir = tmp_path / "staging"
    generate = Mock(return_value="VALUE = 42")
    agent._generate_with_fallback = generate
    monkeypatch.setattr(coder.MCPToolServer, "write_module_to_disk", Mock())
    monkeypatch.setattr(coder.MCPToolServer, "run_pytest_suite", Mock(return_value={"success": True}))
    result = agent.generate_single_module(
        {"target_module": "accounts"}, revision_notes="Change value to 42",
        current_code="VALUE = 1\n# Keep the previous changes",
    )
    assert result["code"] == "VALUE = 42"
    prompt = generate.call_args.args[0]
    assert "Change value to 42" in prompt
    assert "VALUE = 1\n# Keep the previous changes" in prompt
