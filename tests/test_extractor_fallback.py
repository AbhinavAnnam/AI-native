"""Offline checks for provider fallback and complete, schema-valid extraction."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
from google.genai import errors, types

from backend.models.schemas import ProjectSpecs
from phase1_rag import extractor, providers

SPEC = {
    "project_title": "Demo",
    "features": [{"feature_name": "Accounts", "target_module": "accounts",
                  "business_rules": ["Email must be unique"], "functions": [], "exceptions": []}],
    "cross_cutting_constraints": [],
}


@pytest.fixture(autouse=True)
def isolated_config(monkeypatch):
    for key in ("GROQ_API_KEY", "OPENROUTER_API_KEY", "GEMINI_API_KEY",
                "groq_api_key", "openrouter_api_key", "gemini_api_key",
                "GROQ_EXTRACTION_MODEL", "OPENROUTER_EXTRACTION_MODEL"):
        monkeypatch.delenv(key, raising=False)
    extractor.reset_exhausted_models()
    yield
    extractor.reset_exhausted_models()


def response(data, finish_reason="stop"):
    return SimpleNamespace(choices=[SimpleNamespace(
        finish_reason=finish_reason,
        message=SimpleNamespace(content=json.dumps(data)),
    )])


def fake_client(monkeypatch, side_effect):
    client = MagicMock()
    client.__enter__.return_value = client
    client.chat.completions.create.side_effect = side_effect
    factory = Mock(return_value=client)
    monkeypatch.setattr(providers, "OpenAI", factory)
    return factory, client.chat.completions.create


def test_groq_schema_response_does_not_call_gemini(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "offline-key")
    factory, create = fake_client(monkeypatch, [response(SPEC)])
    gemini = SimpleNamespace(models=SimpleNamespace(generate_content=Mock()))
    result = extractor.call_extraction_with_fallback(
        gemini, ["gemini"], "Extract JSON", types.GenerateContentConfig(response_schema=ProjectSpecs),
    )
    assert json.loads(result.text) == SPEC
    gemini.models.generate_content.assert_not_called()
    assert factory.call_args.kwargs["max_retries"] == 0
    request = create.call_args.kwargs
    assert request["model"] == "openai/gpt-oss-20b"
    schema = request["response_format"]["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    assert schema["$defs"]["FeatureSpec"]["additionalProperties"] is False
    assert "additionalProperties" not in ProjectSpecs.model_json_schema()


@pytest.mark.parametrize("bad_response", [response({}, "length"), response({"wrong": "schema"})])
def test_invalid_groq_output_falls_through_to_openrouter(monkeypatch, bad_response):
    monkeypatch.setenv("GROQ_API_KEY", "offline-groq")
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-openrouter")
    factory, create = fake_client(monkeypatch, [bad_response, bad_response, response(SPEC)])
    assert providers.generate_json("Extract JSON", ProjectSpecs) == SPEC
    assert factory.call_args.kwargs["base_url"] == "https://openrouter.ai/api/v1"
    assert create.call_args.kwargs["extra_body"]["provider"]["require_parameters"] is True


def test_provider_auth_failure_skips_other_models_and_uses_openrouter(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "offline-groq")
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-openrouter")
    error = RuntimeError("Provider rejected credentials")
    error.status_code = 401
    _, create = fake_client(monkeypatch, [error, response(SPEC)])
    assert providers.generate_json("Extract JSON", ProjectSpecs) == SPEC
    assert create.call_count == 2


def test_openrouter_works_without_groq_or_gemini_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-openrouter")
    monkeypatch.setenv("OPENROUTER_EXTRACTION_MODEL", "chosen-model")
    _, create = fake_client(monkeypatch, [response(SPEC)])
    result = extractor.call_extraction_with_fallback(
        None, [], "Extract JSON", types.GenerateContentConfig(response_schema=ProjectSpecs),
    )
    assert json.loads(result.text) == SPEC
    assert create.call_args.kwargs["model"] == "chosen-model"


def test_gemini_is_used_when_alternatives_fail(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "offline-groq")
    fake_client(monkeypatch, RuntimeError("Provider unavailable"))
    expected = SimpleNamespace(text=json.dumps(SPEC))
    gemini = SimpleNamespace(models=SimpleNamespace(generate_content=Mock(return_value=expected)))
    assert extractor.call_extraction_with_fallback(
        gemini, ["gemini"], "Extract JSON", types.GenerateContentConfig(),
    ) is expected


def test_gemini_404_still_skips_missing_model():
    expected = SimpleNamespace(text="{}")
    generate = Mock(side_effect=[errors.ClientError(404, {}), expected, expected])
    gemini = SimpleNamespace(models=SimpleNamespace(generate_content=generate))
    for _ in range(2):
        assert extractor.call_gemini_with_fallback(
            gemini, ["missing", "working"], "prompt", types.GenerateContentConfig(),
        ) is expected
    assert [call.kwargs["model"] for call in generate.call_args_list] == ["missing", "working", "working"]


def test_full_map_reduce_works_without_gemini(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "offline-groq")
    _, create = fake_client(monkeypatch, [response({"accounts": "unique emails"}), response(SPEC)])
    monkeypatch.setattr(extractor, "extract_raw_text_from_file_object", lambda file: "Accounts require unique emails.")
    assert extractor.run_extraction_pipeline(b"document", response_schema=ProjectSpecs) == SPEC
    assert create.call_count == 2
    assert create.call_args_list[0].kwargs["response_format"] == {"type": "json_object"}
    assert create.call_args_list[1].kwargs["response_format"]["type"] == "json_schema"


def test_failed_chunk_cannot_produce_partial_success(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "offline-groq")
    monkeypatch.setattr(extractor, "extract_raw_text_from_file_object", lambda file: "x" * 50000)
    generate = Mock(side_effect=[RuntimeError("unavailable"), {"accounts": "unique emails"}])
    monkeypatch.setattr(extractor, "generate_json", generate)
    with pytest.raises(RuntimeError, match="Failed chunks: 1"):
        extractor.run_extraction_pipeline(b"document", response_schema=ProjectSpecs)
    assert generate.call_count == 2


def test_all_provider_failures_report_failure(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "offline-groq")
    fake_client(monkeypatch, RuntimeError("Unavailable"))
    with pytest.raises(RuntimeError, match="Groq/OpenRouter extraction failed"):
        extractor.call_extraction_with_fallback(None, [], "prompt", types.GenerateContentConfig())
