"""JSON extraction through the existing OpenAI-compatible provider clients."""

import json
import logging
import os

from openai import OpenAI

logger = logging.getLogger(__name__)


def fallback_keys_available():
    return any(os.getenv(name) for name in (
        "GROQ_API_KEY", "groq_api_key", "OPENROUTER_API_KEY", "openrouter_api_key",
    ))


def _strict_schema(value):
    """Apply strict object rules to nested objects, including Pydantic $defs."""
    if isinstance(value, list):
        return [_strict_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: _strict_schema(item) for key, item in value.items()}
    if result.get("type") == "object":
        result["additionalProperties"] = False
        result["required"] = list(result.get("properties", {}))
    return result


def generate_json(contents, response_schema=None):
    """Try configured providers once per model; reject incomplete/invalid output."""
    providers = [
        ("Groq", "https://api.groq.com/openai/v1",
         os.getenv("GROQ_API_KEY") or os.getenv("groq_api_key"),
         [os.getenv("GROQ_EXTRACTION_MODEL")] if os.getenv("GROQ_EXTRACTION_MODEL")
         else ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]),
        ("OpenRouter", "https://openrouter.ai/api/v1",
         os.getenv("OPENROUTER_API_KEY") or os.getenv("openrouter_api_key"),
         [os.getenv("OPENROUTER_EXTRACTION_MODEL")] if os.getenv("OPENROUTER_EXTRACTION_MODEL")
         else ["qwen/qwen3.5-35b-a3b", "openai/gpt-oss-20b"]),
    ]
    response_format = {"type": "json_object"}
    if response_schema is not None:
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": response_schema.__name__, "strict": True,
                "schema": _strict_schema(response_schema.model_json_schema()),
            },
        }

    failures = []
    for provider, base_url, api_key, models in providers:
        if not api_key:
            continue
        with OpenAI(api_key=api_key, base_url=base_url, timeout=45.0, max_retries=0) as client:
            for model in models:
                try:
                    logger.info("Extraction request: %s / %s", provider, model)
                    kwargs = {}
                    if provider == "OpenRouter":
                        kwargs["extra_body"] = {"provider": {"require_parameters": True}}
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": (
                                "Extract only facts supported by the supplied document. "
                                "Return a complete JSON object without markdown."
                            )},
                            {"role": "user", "content": contents},
                        ],
                        response_format=response_format,
                        temperature=0.1,
                        max_tokens=8192,
                        **kwargs,
                    )
                    choice = response.choices[0]
                    if choice.finish_reason != "stop" or not choice.message.content:
                        raise ValueError("Model response was empty, refused, or truncated")
                    data = json.loads(choice.message.content)
                    if not isinstance(data, dict) or not data:
                        raise ValueError("Expected a nonempty JSON object")
                    if response_schema is not None:
                        data = response_schema.model_validate(data).model_dump()
                    logger.info("Extraction response validated: %s / %s", provider, model)
                    return data
                except Exception as exc:
                    status = getattr(exc, "status_code", None)
                    reason = str(status) if status else type(exc).__name__
                    failures.append(f"{provider}/{model}: {reason}")
                    logger.warning("Extraction failed: %s / %s (%s)", provider, model, reason)
                    if status in (401, 402, 403):
                        break  # Another model cannot fix this provider's key or credits.
    raise RuntimeError("Groq/OpenRouter extraction failed: " + (
        "; ".join(failures) or "no provider key configured"
    ))
