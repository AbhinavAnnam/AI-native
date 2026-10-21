import logging
import os
import time
import asyncio
import json
from google import genai
from google.genai import types
from typing import List, Literal

from langchain_google_genai import ChatGoogleGenerativeAI

from backend.models.schemas import TaskClassification

logger = logging.getLogger(__name__)


def _gemini_critic_schema(schema):
    """Gemini's OpenAPI schema subset; keep all strict checks in Pydantic locally."""
    original = schema.model_json_schema()
    definitions = original.get("$defs", {})
    def compatible(node):
        if "$ref" in node:
            node = {**definitions[node["$ref"].rsplit("/", 1)[-1]],
                    **{k: v for k, v in node.items() if k != "$ref"}}
        if "anyOf" in node:
            concrete = [v for v in node["anyOf"] if v.get("type") != "null"]
            if len(concrete) == 1 and len(concrete) < len(node["anyOf"]):
                return {**compatible(concrete[0]), "nullable": True}
        result = {k: v for k, v in node.items() if k in {"type", "required", "enum", "description"}}
        if "const" in node:
            result["enum"] = [node["const"]]
        if "properties" in node:
            result["properties"] = {k: compatible(v) for k, v in node["properties"].items()}
        if "items" in node:
            result["items"] = compatible(node["items"])
        return result
    return compatible(original)

# ---------------------------------------------------------------
# Model candidates (priority order). The agent rotates through these
# when a model is overloaded / rate-limited so a transient 429/5xx
# on the provider does not crash the whole request.
# ---------------------------------------------------------------
CLASSIFIER_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.6-flash",
    "gemini-2.5-flash",
    "gemini-3.5-flash-lite",
]
FAST_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
    "gemini-2.5-flash",
]
HEAVY_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-2.5-pro",
]

# Substrings/status codes treated as transient provider errors worth retrying.
TRANSIENT_MARKERS = (
    "429", "500", "502", "503", "504",
    "resource_exhausted", "unavailable", "high demand", "overloaded",
    "rate limit", "quota",
)


def _is_transient(err: Exception) -> bool:
    msg = str(err).lower()
    code = getattr(err, "status_code", None) or getattr(err, "code", None)
    try:
        code = int(code)
    except (TypeError, ValueError):
        code = None
    if code in (429, 500, 502, 503, 504):
        return True
    return any(marker in msg for marker in TRANSIENT_MARKERS)


class _RetryingChatModel:
    """Duck-typed model used by LangGraph nodes (exposes .invoke(messages)).

    Retries transient errors and rotates through all candidate models before raising.
    """

    def __init__(self, candidates: List[str], api_key: str, temperature: float):
        self.candidates = candidates
        self.api_key = api_key
        self.temperature = temperature

    def invoke(self, messages, **kwargs):
        last_err: Exception = None
        for model in self.candidates:
            for attempt in range(2):
                try:
                    llm = ChatGoogleGenerativeAI(
                        model=model,
                        google_api_key=self.api_key,
                        temperature=self.temperature,
                        timeout=60,
                        max_retries=0,
                    )
                    return llm.invoke(messages)
                except Exception as exc:
                    last_err = exc
                    logger.warning(
                        f"[LLMRouter] code model '{model}' failed "
                        f"(attempt {attempt + 1}/2): {exc}"
                    )
                    if _is_transient(exc) and attempt == 0:
                        time.sleep(1.0)
                        continue
                    break
        raise last_err


class LLMRouter:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY") or ""

    def _run_with_fallback(self, candidates, temperature, prompt, response_schema=None):
        last_err: Exception = None
        for model in candidates:
            for attempt in range(2):
                try:
                    llm = ChatGoogleGenerativeAI(
                        model=model,
                        google_api_key=self.api_key,
                        temperature=temperature,
                        timeout=30,
                        max_retries=0,
                    )
                    if response_schema is not None:
                        llm = llm.with_structured_output(response_schema)
                    return llm.invoke(prompt)
                except Exception as exc:
                    last_err = exc
                    logger.warning(
                        f"[LLMRouter] classifier '{model}' failed "
                        f"(attempt {attempt + 1}/2): {exc}"
                    )
                    if _is_transient(exc) and attempt == 0:
                        time.sleep(1.0)
                        continue
                    break
        raise last_err

    def route(self, prompt: str) -> TaskClassification:
        classification_prompt = (
            "Analyze the following user task or spec directive and classify its complexity.\n"
            "Simple queries or formatting tasks should use 'fast'. Complex multi-file code generation, "
            "refactoring, or architectural reasoning must use 'heavy'.\n\n"
            f"Task: {prompt}"
        )
        return self._run_with_fallback(
            CLASSIFIER_MODELS,
            0.0,
            classification_prompt,
            response_schema=TaskClassification,
        )

    def get_model(self, tier: Literal["fast", "heavy"]):
        if tier == "fast":
            return _RetryingChatModel(FAST_MODELS, self.api_key, temperature=0.2)
        return _RetryingChatModel(HEAVY_MODELS, self.api_key, temperature=0.1)

    def review_json(self, prompt, schema, max_output_tokens, validator=None, max_attempts=2):
        """Bounded critic fallback using native JSON output, without tool calling."""
        if not self.api_key:
            raise RuntimeError("Gemini fallback is not configured")
        from phase1_rag.providers import critic_gemini_models
        candidates = critic_gemini_models()
        last_error = None
        from phase1_rag.providers import (ReviewTimeLimitError, ProviderRateLimitError,
            critic_request_timeout, check_critic_quota, remember_critic_quota,
            critic_provider_ready, remember_critic_failure, ProviderUnavailableError)
        ready = [m for m in dict.fromkeys(candidates) if critic_provider_ready(self.api_key, "Gemini", m)]
        if not ready:
            raise ProviderUnavailableError()
        attempts = [(model, prompt) for model in ready[:max_attempts]]
        correction_used = False
        for index, (model, request_prompt) in enumerate(attempts):
            try:
                if not critic_provider_ready(self.api_key, "Gemini", model):
                    last_error = last_error or ProviderUnavailableError()
                    continue
                check_critic_quota(self.api_key, "Gemini", model)
                logger.info("Critic request: Gemini / %s", model)
                # Use the proven native SDK path. LangChain's schema wrapper
                # produces INVALID_ARGUMENT for this critic schema/model pair.
                timeout = critic_request_timeout()
                async def request():
                    async with genai.Client(api_key=self.api_key, http_options=types.HttpOptions(
                        timeout=int(timeout * 1000), retry_options=types.HttpRetryOptions(attempts=1))).aio as client:
                        result = await client.models.generate_content(model=model, contents=request_prompt,
                            config=types.GenerateContentConfig(response_mime_type="application/json",
                                response_schema=_gemini_critic_schema(schema),
                                # Leave space for both the JSON report and model
                                # overhead; the caller's local schema stays strict.
                                max_output_tokens=min(8192, max_output_tokens + 2048),
                                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)))
                        if not result.candidates or str(result.candidates[0].finish_reason).split(".")[-1] != "STOP":
                            raise ValueError("Incomplete critic response")
                        return schema.model_validate_json(result.text or "").model_dump()
                async def bounded():
                    return await asyncio.wait_for(request(), timeout=timeout)
                data = asyncio.run(bounded())
                if validator is not None:
                    validator(data)
                logger.info("Critic response validated: Gemini / %s", model)
                return data
            except Exception as exc:
                if isinstance(exc, ReviewTimeLimitError):
                    raise
                if isinstance(exc, ProviderRateLimitError):
                    raise
                if (getattr(exc, "status_code", None) or getattr(exc, "code", None)) == 429:
                    logger.warning("Critic quota reached: Gemini / %s; preserving completed batches", model)
                    raise remember_critic_quota(self.api_key, "Gemini", model, exc) from exc
                critic_request_timeout()
                remember_critic_failure(self.api_key, "Gemini", model, exc)
                last_error = exc
                logger.warning("Critic failed: Gemini / %s (%s) status=%s", model, type(exc).__name__,
                               getattr(exc, "status_code", None) or getattr(exc, "code", None) or "unknown")
                status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
                if isinstance(status, int) and 500 <= status <= 599:
                    last_error = ProviderUnavailableError()
                    if not correction_used:
                        alternate = next((m for m in candidates if m != model and critic_provider_ready(self.api_key, "Gemini", m)), None)
                        if alternate:
                            correction_used = True
                            attempts.insert(index + 1, (alternate, prompt))
                if getattr(exc, "critic_issues", None):
                    from phase1_rag.providers import _safe_validation_issues
                    issues = _safe_validation_issues(exc, schema)
                    logger.warning("Critic consistency checks failed: %s", json.dumps(issues))
                    if not correction_used:
                        correction_used = True
                        corrected_prompt = prompt + "\nCorrect these report consistency errors: " + json.dumps(issues) + (
                            ". Return every supplied requirement ID exactly once. Cite only supplied source paths. "
                            "Use null for uncertain line numbers. Return the complete corrected report.")
                        attempts.insert(index + 1, (model, corrected_prompt))
        raise last_error
