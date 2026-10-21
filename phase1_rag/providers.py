"""JSON extraction through the existing OpenAI-compatible provider clients."""

import json
import logging
import os
import re
import math
import time
import asyncio
from contextlib import nullcontext
from contextvars import ContextVar
from types import SimpleNamespace
from threading import Lock

from openai import AsyncOpenAI, OpenAI
from pydantic import ValidationError

logger = logging.getLogger(__name__)
_critic_quotas = {}
_critic_size_limits = {}
_critic_json_modes = {}
_critic_failures = {}
critic_deadline = ContextVar("critic_deadline", default=None)
critic_provider_cooldowns = ContextVar("critic_provider_cooldowns", default=None)
_critic_groq_lock = Lock()
NVIDIA_CRITIC_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"


def critic_groq_limit():
    return min(7500, max(1024, int(os.getenv("CRITIC_GROQ_REQUEST_TOKEN_LIMIT", "7500"))))


def critic_openrouter_model(*, default_model="openrouter/free"):
    """Extraction overrides must never select a paid route for the critic."""
    if default_model != "openrouter/free" and not re.fullmatch(r"[\w.-]+/[\w.-]+:free", default_model):
        raise ValueError("The critic default must be a free route.")
    model = os.getenv("CRITIC_OPENROUTER_MODEL") or default_model
    if model == "openrouter/free" or re.fullmatch(r"[\w.-]+/[\w.-]+:free", model):
        return model
    logger.warning("Ignoring a non-free CRITIC_OPENROUTER_MODEL; using %s", default_model)
    return default_model


def critic_gemini_models():
    # Free-tier eligible models; the project's Google billing tier still applies.
    allowed = {"gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-2.5-flash-lite", "gemini-2.5-flash"}
    model = os.getenv("CRITIC_GEMINI_MODEL", "gemini-3.5-flash-lite")
    if model not in allowed:
        logger.warning("Ignoring a critic Gemini override without the selected free-tier policy")
        model = "gemini-3.5-flash-lite"
    return list(dict.fromkeys([model, "gemini-3.1-flash-lite"]))


def critic_token_estimate(messages, max_tokens, response_format=None):
    """Conservative byte estimate, including the actual schema and output reserve."""
    size = len(json.dumps({"messages": messages, "response_format": response_format}, ensure_ascii=False).encode())
    return math.ceil(size / 3) + max_tokens + 256


def critic_response_format(schema):
    return ({"type": "json_schema", "json_schema": {"name": schema.__name__, "strict": True,
            "schema": _strict_schema(schema.model_json_schema())}} if schema else {"type": "json_object"})


class ReviewTimeLimitError(RuntimeError):
    pass


class ProviderUnavailableError(RuntimeError):
    def __init__(self, retry_after=10):
        self.retry_after = retry_after
        super().__init__("Model providers are temporarily unavailable. Completed work is saved; retry the same action to resume.")


class ProviderResponseError(RuntimeError):
    def __init__(self, issues=None):
        self.issues = (issues or [{"field": "report", "category": "invalid_response"}])[:12]
        super().__init__("The critic returned an invalid report. Your drafts and completed review work are saved. No automatic retry was started.")


class ProviderBillingError(RuntimeError):
    def __init__(self, provider="OpenRouter", limit_source=None):
        self.limit_source = limit_source if limit_source in {"openrouter_in_flight_budget", "openrouter_key_limit", "openrouter_credits"} else None
        detail = ("Its in-flight spending budget is full; wait for active requests to finish."
                  if self.limit_source == "openrouter_in_flight_budget" else "Check the provider's credits and API-key spending limits.")
        super().__init__(f"{provider} rejected the request with HTTP 402 (credit or spending-budget limit). {detail} Your code and completed work are saved; no automatic retry was started.")


def _safe_validation_issues(exc, schema):
    fields = set()
    def visit(node):
        if isinstance(node, dict):
            fields.update(node.get("properties", {}))
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)
    if schema is not None:
        visit(schema.model_json_schema())
    declared = getattr(exc, "critic_issues", None)
    if isinstance(declared, list) and declared:
        return [{"field": ".".join(part if part in fields or part.isdigit() else "<unexpected_field>"
                for part in str(issue.get("field", "report")).split(".")),
                "category": issue["category"] if re.fullmatch(r"[a-z_]{1,80}", str(issue.get("category", ""))) else "invalid_response"}
                for issue in declared[:12] if isinstance(issue, dict)]
    if isinstance(exc, ValidationError):
        return [{"field": ".".join(str(part) if isinstance(part, int) or part in fields else "<unexpected_field>"
                                    for part in error["loc"]), "category": error["type"]}
                for error in exc.errors(include_input=False, include_url=False)][:12]
    return [{"field": "report", "category": "invalid_json" if isinstance(exc, json.JSONDecodeError) else "invalid_response"}]


def critic_request_timeout(*, model=None):
    limit = max(1, float(os.getenv("CRITIC_MODEL_TIMEOUT_SECONDS", "45")))
    if model == NVIDIA_CRITIC_MODEL:
        # A reasoning review can outlast the ordinary provider deadline. Keep
        # both a hard per-request ceiling and the existing whole-review budget.
        limit = min(120, max(1, float(os.getenv("CRITIC_NVIDIA_TIMEOUT_SECONDS", "90"))))
    deadline = critic_deadline.get()
    if deadline is not None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ReviewTimeLimitError("Review time budget reached. Completed work is saved; Run Critic again to continue.")
        limit = min(limit, remaining)
    return limit


def critic_nvidia_output_tokens(report_tokens):
    """Allow room for Ultra's observed reasoning without shrinking the report.

    A real 4,864-token report exhausted 5,888 total tokens after 3,613 thinking
    tokens. Budget at least 6,144 tokens of reasoning headroom, growing with
    report size, within a separate bounded 16,384-token completion ceiling.
    """
    return min(16384, max(8192, report_tokens + max(6144, report_tokens)))


def _critic_request(api_key, base_url, *, _json_review=False, **kwargs):
    """Cancel the socket at a wall-clock deadline, including trickling responses."""
    async def request():
        async with AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0) as client:
            raw = await client.chat.completions.with_raw_response.create(**kwargs)
            parsed = raw.parse()
            return SimpleNamespace(headers=raw.headers, parse=lambda: parsed)
    groq = base_url == "https://api.groq.com/openai/v1"
    required = critic_token_estimate(kwargs.get("messages", []), kwargs.get("max_tokens", 0), kwargs.get("response_format"))
    if groq and required > critic_groq_limit():
        raise ValueError("Critic request exceeds the Groq token budget; use a free fallback.")
    if groq and not _critic_groq_lock.acquire(timeout=critic_request_timeout()):
        raise ProviderRateLimitError(10)
    try:
        quota_key = (hash(api_key), "Groq", kwargs.get("model"))
        if groq:
            _quota_wait(quota_key, required, 0)
            quota = _critic_quotas.get(quota_key)
            # Reserve even when the provider omits headers or a call times out.
            _critic_quotas[quota_key] = {"remaining": max(0, (quota["remaining"] if quota else critic_groq_limit()) - required),
                                       "retry_at": quota["retry_at"] if quota else time.time() + 60}
        timeout = critic_request_timeout(model=kwargs.get("model") if kwargs.get("response_format") or _json_review else None)
        try:
            raw = asyncio.run(asyncio.wait_for(request(), timeout=timeout))
        except TimeoutError:
            critic_request_timeout()
            raise
        except Exception as exc:
            if groq and getattr(exc, "status_code", None) == 429:
                headers = getattr(getattr(exc, "response", None), "headers", {})
                _critic_quotas[quota_key] = {"remaining": 0, "retry_at": time.time() + _seconds(headers.get("retry-after"))}
            raise
        if groq and raw.headers.get("x-ratelimit-remaining-tokens") is not None:
            _critic_quotas[quota_key] = {"remaining": float(raw.headers["x-ratelimit-remaining-tokens"]),
                "retry_at": time.time() + _seconds(raw.headers.get("x-ratelimit-reset-tokens"))}
        if groq:
            raw._critic_quota_recorded = True
        return raw
    finally:
        if groq:
            _critic_groq_lock.release()


class ProviderRateLimitError(RuntimeError):
    def __init__(self, retry_after=60):
        self.retry_after = max(1, min(float(retry_after), 86400))
        super().__init__("Provider quota reached. Completed critic batches are preserved; resume after the indicated wait.")


def _seconds(value, default=60):
    try:
        duration = float(value)
    except (ValueError, TypeError):
        units = {"ms": .001, "s": 1, "m": 60, "h": 3600}
        parts = re.findall(r"(\d+(?:\.\d+)?)(ms|s|m|h)", str(value or ""))
        duration = sum(float(n) * units[u] for n, u in parts) if parts else default
    return max(0, min(duration, 86400)) if math.isfinite(duration) else default


def _quota_wait(key, required_tokens, waited):
    quota = _critic_quotas.get(key)
    if not quota or quota["remaining"] >= required_tokens:
        return waited
    delay = max(0, quota["retry_at"] - time.time())
    if delay > max(0, float(os.getenv("CRITIC_MAX_WAIT_SECONDS", "20")) - waited):
        raise ProviderRateLimitError(delay)
    if delay:
        if delay >= critic_request_timeout():
            raise ProviderRateLimitError(delay)
        logger.info("Critic waiting %.1f seconds for provider quota", delay)
        time.sleep(delay)
    _critic_quotas.pop(key, None)
    return waited + delay


def check_critic_quota(api_key, provider, model):
    """Reuse critic quota reservations for native SDK fallbacks as well."""
    return _quota_wait((hash(api_key), provider, model), 1, 0)


def critic_provider_ready(api_key, provider, model):
    key = (hash(api_key), provider, model)
    return _critic_failures.get(key, 0) <= time.time()


def critic_quota_ready(api_key, provider, model):
    quota = _critic_quotas.get((hash(api_key), provider, model))
    return not quota or quota["remaining"] > 0 or quota["retry_at"] <= time.time()


def remember_critic_failure(api_key, provider, model, exc):
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    slow = isinstance(exc, TimeoutError) or type(exc).__name__ in {"APITimeoutError", "ReadTimeout"}
    truncated = isinstance(exc, ValueError) and str(exc) in {"truncated_response", "Incomplete critic response", "Incomplete repair", "Truncated repair"}
    server_error = (isinstance(status, int) and 500 <= status <= 599) or type(exc).__name__ == "ServerError"
    if slow or truncated or server_error or status in {400, 401, 403, 404}:
        _critic_failures[(hash(api_key), provider, model)] = time.time() + (30 if server_error else 600 if slow else 300)


def remember_critic_quota(api_key, provider, model, exc):
    """Keep only a retry deadline, never the provider response or input data."""
    headers = getattr(getattr(exc, "response", None), "headers", {}) or {}
    delays = [_seconds(headers.get("retry-after"))] if headers.get("retry-after") else []
    body = getattr(exc, "response_json", None)
    error = body.get("error", {}) if isinstance(body, dict) else {}
    details = error.get("details", []) if isinstance(error, dict) else []
    for detail in details if isinstance(details, list) else []:
        if isinstance(detail, dict) and str(detail.get("@type", "")).endswith("/google.rpc.RetryInfo"):
            delays.append(_seconds(detail.get("retryDelay")))
    delay = max(delays, default=60)
    _critic_quotas[(hash(api_key), provider, model)] = {"remaining": 0, "retry_at": time.time() + delay}
    return ProviderRateLimitError(delay)


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


def _safe_error_identifiers(exc, api_key):
    """Log bounded identifiers only; provider messages may echo prompts or source."""
    body = getattr(exc, "body", None)
    error = body.get("error", body) if isinstance(body, dict) else {}
    error = error if isinstance(error, dict) else {}
    details = {}
    for field in ("code", "type", "param"):
        value = error.get(field) or getattr(exc, field, None)
        if value is None:
            details[field] = "-"
        elif (isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.\[\]-]{1,100}", value)
              and not (api_key and api_key in value) and not value.startswith(("gsk_", "sk-", "AIza"))):
            details[field] = value
        else:
            details[field] = "[redacted]"
    return details


def generate_json(contents, response_schema=None, *, system_prompt=None, max_tokens=None, max_attempts=None, critic_mode=False, validator=None, prefer_independent_fallback=False):
    """Try configured providers once per model; reject incomplete/invalid output."""
    max_tokens = max_tokens if max_tokens is not None else (4096 if critic_mode else 8192)
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
    if critic_mode:
        # Critic-only policy; extraction's paid model overrides are independent.
        models = [os.getenv("CRITIC_GROQ_MODEL")] if os.getenv("CRITIC_GROQ_MODEL") else ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
        name, url, key, _ = providers[0]
        providers[0] = (name, url, key, models)
        name, url, key, _ = providers[1]
        # Prefer an explicit reasoning model over the random free router. The
        # existing generic route remains within the same total attempt budget.
        # Repair calls keep their separate existing default via the helper above.
        providers[1] = (name, url, key, list(dict.fromkeys([
            critic_openrouter_model(default_model=NVIDIA_CRITIC_MODEL),
            NVIDIA_CRITIC_MODEL, "openrouter/free"])))
        if prefer_independent_fallback:
            name, url, key, models = providers[1]
            providers[1] = (name, url, key, models[:1])
        max_attempts = 4 if max_attempts is None else max_attempts
    label = "Critic" if critic_mode else "Extraction"
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
    attempts, waited = 0, 0
    quota_errors = []
    response_issues = []
    billing_errors = []
    schema_bytes = len(json.dumps(response_format).encode())
    input_bytes = len(contents.encode())
    estimated_tokens = (input_bytes + schema_bytes + len((system_prompt or "").encode())) // 3 + max_tokens
    if critic_mode:
        estimated_tokens = critic_token_estimate([{"role": "system", "content": system_prompt or ""},
            {"role": "user", "content": contents}], max_tokens, response_format)
    larger_review = estimated_tokens > critic_groq_limit()
    for provider, base_url, api_key, models in providers:
        if not api_key:
            continue
        if critic_mode and provider == "Groq" and larger_review:
            logger.info("Critic routing larger review directly to fallback providers (estimated tokens=%s)", estimated_tokens)
            continue
        output_tokens = max_tokens
        if not critic_mode and provider == "Groq":
            # Phase 1's former 8192-token output cap alone exceeded an 8000 TPM plan.
            # Count the entire envelope, leaving at least 2048 tokens for complete JSON.
            limit = min(7500, max(2048, int(os.getenv("EXTRACTION_GROQ_REQUEST_TOKEN_LIMIT", "7500"))))
            prompt = system_prompt or "Extract only facts supported by the supplied document. Return a complete JSON object without markdown."
            required_input = critic_token_estimate([{"role": "system", "content": prompt},
                {"role": "user", "content": contents}], 0, response_format)
            available = limit - required_input
            if available < 2048:
                logger.info("Extraction skipping oversized Groq request (estimated input=%s, limit=%s, minimum output=2048)", required_input, limit)
                continue
            output_tokens = min(max_tokens, available)
        use_json_object = False
        with (nullcontext(None) if critic_mode else OpenAI(api_key=api_key, base_url=base_url, timeout=45.0, max_retries=0)) as client:
            for model in models:
                quota_key = (hash(api_key), provider, model)
                if critic_mode and not critic_provider_ready(api_key, provider, model):
                    logger.info("Critic skipping cooling provider: %s / %s", provider, model)
                    continue
                cooldowns = critic_provider_cooldowns.get() if critic_mode else None
                if cooldowns is not None and cooldowns.get(quota_key, 0) > time.time():
                    logger.info("Critic skipping temporarily unavailable %s / %s for remaining batches", provider, model)
                    continue
                size_limit = _critic_size_limits.get(quota_key, {})
                if critic_mode and size_limit.get("expires", 0) > time.time() and input_bytes >= size_limit["bytes"]:
                    continue
                mode = {"type": "json_object"} if critic_mode and (use_json_object or
                       _critic_json_modes.get(quota_key, 0) > time.time()) else response_format
                # Ultra's active free endpoint does not advertise response_format.
                # Supply the schema in the prompt and enforce it strictly below.
                prompt_only = critic_mode and provider == "OpenRouter" and model == NVIDIA_CRITIC_MODEL
                if prompt_only:
                    mode = {"type": "json_object"}
                model_estimated_tokens = estimated_tokens
                if prompt_only:
                    model_estimated_tokens += max(0, critic_nvidia_output_tokens(max_tokens) - max_tokens)
                stop_provider = False
                while max_attempts is None or attempts < max_attempts:
                    quota_key = (hash(api_key), provider, model)  # Process-local, no credentials stored.
                    try:
                        if critic_mode:
                            waited = _quota_wait(quota_key, model_estimated_tokens, waited)
                        attempts += 1
                        logger.info("%s request: %s / %s", label, provider, model)
                        kwargs = {}
                        if provider == "OpenRouter":
                            kwargs["extra_body"] = {"provider": {"require_parameters": True}}
                            if critic_mode:
                                # A free router can choose different model families.
                                kwargs["extra_body"]["reasoning"] = {"exclude": True}
                                if model == NVIDIA_CRITIC_MODEL:
                                    kwargs["reasoning_effort"] = "low"
                                    output_tokens = critic_nvidia_output_tokens(max_tokens)
                                    logger.info("Critic Ultra output allowance: report=%s reasoning_headroom=%s total=%s",
                                                max_tokens, output_tokens - max_tokens, output_tokens)
                                else:
                                    output_tokens = max_tokens
                        elif critic_mode and model.startswith("openai/gpt-oss-"):
                            kwargs["reasoning_effort"] = "low"
                            kwargs["extra_body"] = {"include_reasoning": False}
                        prompt = system_prompt or (
                            "Extract only facts supported by the supplied document. Return a complete JSON object without markdown."
                        )
                        if critic_mode and mode["type"] == "json_object" and response_schema is not None:
                            prompt += "\nRequired JSON schema: " + json.dumps(response_format["json_schema"]["schema"], separators=(",", ":"))
                        if prompt_only:
                            prompt += "\nReturn only a complete JSON object. Do not use Markdown fences or explanatory text."
                        messages = [{"role": "system", "content": prompt}, {"role": "user", "content": contents}]
                        if critic_mode and provider == "Groq" and critic_token_estimate(messages, max_tokens, mode) > critic_groq_limit():
                            break
                        create = (lambda **args: _critic_request(api_key, base_url, _json_review=True, **args)) if critic_mode else client.chat.completions.create
                        response = create(
                            model=model,
                            messages=messages,
                            temperature=0.1,
                            max_tokens=output_tokens,
                            **({"response_format": mode} if not prompt_only else {}),
                            **kwargs,
                        )
                        if critic_mode:
                            headers = response.headers
                            remaining = headers.get("x-ratelimit-remaining-tokens")
                            if remaining is not None and not getattr(response, "_critic_quota_recorded", False):
                                _critic_quotas[quota_key] = {"remaining": float(remaining),
                                    "retry_at": time.time() + _seconds(headers.get("x-ratelimit-reset-tokens"))}
                            response = response.parse()
                        choice = response.choices[0]
                        if choice.finish_reason != "stop" or not choice.message.content:
                            # Fixed diagnostic categories; never log provider text or source.
                            reason = "truncated_response" if choice.finish_reason == "length" else "empty_or_refused_response"
                            logger.warning("%s response rejected: %s", label, reason)
                            if critic_mode:
                                usage = getattr(response, "usage", None)
                                details = getattr(usage, "completion_tokens_details", None)
                                logger.warning("Critic output budget: cap=%s completion=%s reasoning=%s", output_tokens,
                                    getattr(usage, "completion_tokens", None), getattr(details, "reasoning_tokens", None))
                            raise ValueError(reason)
                        data = json.loads(choice.message.content)
                        if not isinstance(data, dict) or not data:
                            raise ValueError("Expected a nonempty JSON object")
                        if response_schema is not None:
                            data = response_schema.model_validate(data).model_dump()
                        if validator is not None:
                            validator(data)
                        logger.info("%s response validated: %s / %s", label, provider, model)
                        return data
                    except ReviewTimeLimitError:
                        raise
                    except ProviderRateLimitError as exc:
                        quota_errors.append(exc)
                        stop_provider = True
                        break
                    except Exception as exc:
                        if critic_mode:
                            remember_critic_failure(api_key, provider, model, exc)
                        status = getattr(exc, "status_code", None)
                        reason = str(status) if status else type(exc).__name__
                        failures.append(f"{provider}/{model}: {reason}")
                        details = _safe_error_identifiers(exc, api_key)
                        if cooldowns is not None and (isinstance(exc, TimeoutError) or
                            type(exc).__name__ in {"APITimeoutError", "ReadTimeout"} or
                            (isinstance(exc, ValueError) and str(exc) == "truncated_response")):
                            # Other batches can immediately use a healthy fallback.
                            # This cache belongs to this review; nothing is disabled permanently.
                            cooldowns[quota_key] = time.time() + 300
                        logger.warning(
                            "%s failed: %s / %s (%s) schema=%s code=%s type=%s param=%s",
                            label, provider, model, reason, response_schema.__name__ if response_schema else "json_object",
                            details["code"], details["type"], details["param"],
                        )
                        if isinstance(exc, ValidationError):
                            logger.warning("%s schema validation categories: %s", label,
                                           ",".join(sorted({e["type"] for e in exc.errors(include_input=False, include_url=False)})))
                        if critic_mode and (isinstance(exc, ValueError) or details["code"] == "json_validate_failed"):
                            issues = _safe_validation_issues(exc, response_schema)
                            response_issues.extend(issues)
                            logger.warning("Critic invalid report fields: %s", json.dumps(issues, separators=(",", ":")))
                        if critic_mode and status == 429:
                            headers = getattr(getattr(exc, "response", None), "headers", {})
                            delay = _seconds(headers.get("retry-after") or headers.get("x-ratelimit-reset-tokens"))
                            _critic_quotas[quota_key] = {"remaining": 0, "retry_at": time.time() + delay}
                            try:
                                if attempts >= max_attempts:
                                    raise ProviderRateLimitError(delay)
                                waited = _quota_wait(quota_key, 1, waited)
                            except ProviderRateLimitError as pause:
                                quota_errors.append(pause)
                                stop_provider = True
                                break
                            continue
                        if critic_mode and status == 413:
                            _critic_size_limits[quota_key] = {"bytes": input_bytes, "expires": time.time() + 600}
                            break
                        if critic_mode and status == 402:
                            body = getattr(exc, "body", {}) or {}
                            error = body.get("error", body) if isinstance(body, dict) else {}
                            metadata = error.get("metadata", {}) if isinstance(error, dict) else {}
                            billing_errors.append(ProviderBillingError(provider, metadata.get("limit_source") if isinstance(metadata, dict) else None))
                            stop_provider = True
                            break
                        if critic_mode and mode["type"] == "json_schema" and (
                            details["code"] == "json_validate_failed" or isinstance(exc, ValueError)
                        ):
                            # Spend the next attempt on the next model with a simpler wire format.
                            # Local schema and semantic validation remain mandatory.
                            use_json_object = True
                            if details["code"] == "json_validate_failed":
                                _critic_json_modes[quota_key] = time.time() + 600
                                # GPT-OSS can satisfy the local report contract in
                                # JSON-object mode even when native generation
                                # rejects its schema output. Spend one existing
                                # attempt on that model before rotating away.
                                mode = {"type": "json_object"}
                                continue
                            if len(models) == 1:
                                mode = {"type": "json_object"}
                                continue
                        stop_provider = status in (401, 402, 403)
                        break
                if stop_provider:
                    break
    if response_issues:
        raise ProviderResponseError(response_issues)
    if billing_errors:
        raise billing_errors[0]
    if quota_errors:
        raise ProviderRateLimitError(min(e.retry_after for e in quota_errors))
    raise RuntimeError("Groq/OpenRouter extraction failed: " + (
        "; ".join(failures) or "no provider key configured"
    ))
