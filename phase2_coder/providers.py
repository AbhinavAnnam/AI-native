"""Bounded draft generation with expiring provider cooldowns and free fallbacks."""
import ast
import asyncio
import json
import logging
import math
import os
import re
import time
from decimal import Decimal
from threading import Lock
from types import SimpleNamespace

import httpx
from google import genai
from google.genai import types
from openai import APITimeoutError, AsyncOpenAI

logger = logging.getLogger(__name__)
DEEPSEEK_FREE = "deepseek/deepseek-v4-flash:free"
FREE_CODER = "cohere/north-mini-code:free"
NEMOTRON_FREE = "nvidia/nemotron-3-ultra-550b-a55b:free"
_cooldowns, _quotas, _availability = {}, {}, {}
_state_lock = Lock()


class DraftGenerationError(RuntimeError):
    """All currently usable providers failed within the bounded draft budget."""


class DraftResponseError(ValueError):
    """A fixed diagnostic for an unusable response, not a provider outage."""

    def __init__(self, category):
        self.category = category
        super().__init__(f"Incomplete draft ({category})")


def _seconds(value, default=60):
    try:
        duration = float(value)
    except (ValueError, TypeError):
        parts = re.findall(r"(\d+(?:\.\d+)?)(ms|s|m|h)", str(value or ""))
        duration = sum(float(n) * {"ms": .001, "s": 1, "m": 60, "h": 3600}[unit]
                       for n, unit in parts) if parts else default
    return max(0, min(duration, 86400)) if math.isfinite(duration) else default


def _input_tokens(messages):
    return math.ceil(len(json.dumps(messages, ensure_ascii=False).encode()) / 3) + 256


def _free_model(model):
    return model == "openrouter/free" or bool(re.fullmatch(r"[\w.-]+/[\w.-]+:free", model))


def _free_available(model):
    """Cache public endpoint discovery briefly; unavailable routes can return."""
    if not _free_model(model):
        return False
    if model == "openrouter/free":
        return True
    now = time.monotonic()
    with _state_lock:
        cached = _availability.get(model)
    if cached and cached[0] > now:
        return cached[1]
    available = False
    try:
        response = httpx.get(f"https://openrouter.ai/api/v1/models/{model}/endpoints", timeout=4)
        response.raise_for_status()
        endpoints = response.json()["data"]["endpoints"]
        available = any(all(Decimal(str(endpoint.get("pricing", {}).get(field, "1"))) == 0
                            for field in ("prompt", "completion")) for endpoint in endpoints)
    except Exception:
        logger.info("Coder could not verify free endpoints for %s; will recheck later", model)
    with _state_lock:
        _availability[model] = (time.monotonic() + 60, available)
    return available


def _candidates(keys):
    groq = list(dict.fromkeys(m.strip() for m in os.getenv("CODER_GROQ_MODELS",
        "openai/gpt-oss-20b,openai/gpt-oss-120b").split(",") if m.strip()))
    if keys.get("groq"):
        for model in groq:
            yield "Groq", model, keys["groq"]
    if keys.get("gemini"):
        yield "Gemini", "gemini-3.5-flash-lite", keys["gemini"]
    if keys.get("openrouter"):
        yield "OpenRouter", NEMOTRON_FREE, keys["openrouter"]
    if keys.get("openrouter"):
        override = os.getenv("CODER_OPENROUTER_MODEL", "")
        if override and not _free_model(override):
            logger.warning("Ignoring paid CODER_OPENROUTER_MODEL; only free draft routes are allowed")
            override = ""
        for model in dict.fromkeys(filter(None, [override, DEEPSEEK_FREE, FREE_CODER, "openrouter/free"])):
            if model != NEMOTRON_FREE:
                yield "OpenRouter", model, keys["openrouter"]


def _request(*, provider, model, key, messages, max_tokens, timeout):
    """Cancel the network operation at a wall-clock timeout; SDK retries are off."""
    async def perform():
        if provider == "Gemini":
            async with genai.Client(api_key=key, http_options=types.HttpOptions(
                timeout=int(timeout * 1000), retry_options=types.HttpRetryOptions(attempts=1))).aio as client:
                response = await client.models.generate_content(model=model,
                    contents="\n\n".join(m["content"] for m in messages),
                    config=types.GenerateContentConfig(temperature=.1, max_output_tokens=max_tokens,
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)))
                if not response.candidates or str(response.candidates[0].finish_reason).split(".")[-1] != "STOP":
                    raise DraftResponseError("truncated_or_refused_response")
                return SimpleNamespace(code=response.text, headers={})
        url = "https://api.groq.com/openai/v1" if provider == "Groq" else "https://openrouter.ai/api/v1"
        async with AsyncOpenAI(api_key=key, base_url=url, timeout=timeout, max_retries=0) as client:
            extra = {"reasoning_effort": "low", "extra_body": {"include_reasoning": False}} if (
                provider == "Groq" and model.startswith("openai/gpt-oss-")) else {}
            if provider == "OpenRouter":
                extra = {"extra_body": {"reasoning": {"exclude": True}}}
                if model == NEMOTRON_FREE:
                    # Ultra supports native effort control. Avoid numeric thinking
                    # limits unsupported by its current free provider adapter.
                    extra["reasoning_effort"] = "low"
                    max_tokens_with_reasoning = max_tokens + 1024
                else:
                    max_tokens_with_reasoning = max_tokens
            else:
                max_tokens_with_reasoning = max_tokens
            raw = await client.chat.completions.with_raw_response.create(model=model,
                messages=messages, temperature=.1, max_tokens=max_tokens_with_reasoning, **extra)
            choices = raw.parse().choices
            if not choices:
                raise DraftResponseError("empty_response")
            choice = choices[0]
            if choice.finish_reason != "stop":
                raise DraftResponseError("truncated_response" if choice.finish_reason == "length" else "refused_response")
            return SimpleNamespace(code=choice.message.content, headers=raw.headers)
    async def bounded():
        return await asyncio.wait_for(perform(), timeout=timeout)
    return asyncio.run(bounded())


def _route_health(provider, model, key):
    with _state_lock:
        until, waitable = _cooldowns.get((hash(key), provider, model), (0, False))
        provider_until = _cooldowns.get((hash(key), provider, "*"), (0, False))[0]
    return (provider_until, False) if provider_until > until else (until, waitable)


def generate_draft_text(prompt, system_prompt, *, keys, budget=None):
    """Full requirements, one targeted syntax correction, one shared call/time budget."""
    budget = {} if budget is None else budget
    deadline = budget.setdefault("deadline", time.monotonic() + max(1, float(os.getenv("CODER_MODULE_TIMEOUT_SECONDS", "120"))))
    timeout = max(1, float(os.getenv("CODER_MODEL_TIMEOUT_SECONDS", "45")))
    attempt_limit = max(1, min(8, int(os.getenv("CODER_MAX_PROVIDER_ATTEMPTS", "4"))))
    wait_limit = max(0, min(10, float(os.getenv("CODER_MAX_WAIT_SECONDS", "10"))))
    if sum(bool(keys.get(name)) for name in ("groq", "gemini", "openrouter")) > 1:
        wait_limit = 0
    attempts, waited = budget.get("attempts", 0), budget.get("waited", 0)
    base_messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
    pending, deferred, rechecked, tried = list(_candidates(keys)), {}, set(), set()
    gemini_route = ("Gemini", "gemini-3.5-flash-lite", keys.get("gemini"))
    while pending or deferred:
        if attempts >= attempt_limit or time.monotonic() >= deadline:
            break
        # A slow fallback can outlast another provider's cooldown. Revisit each
        # previously skipped route at most once, without sleeping or new budgets.
        recovered = [route for route, until in deferred.items()
                     if max(until, _route_health(*route)[0]) <= time.monotonic()]
        for route in recovered:
            deferred.pop(route)
            rechecked.add(route)
        pending[:0] = recovered
        if not pending:
            break
        provider, model, key = route = pending.pop(0)
        state_key = (hash(key), provider, model)
        until, waitable = _route_health(*route)
        delay = max(0, until - time.monotonic())
        if delay and (not waitable or delay + .05 > wait_limit - waited):
            logger.info("Coder cooling down %s / %s for %.1fs; eligible automatically afterward", provider, model, delay)
            if route not in rechecked:
                deferred[route] = until
            continue
        gemini_pending = bool(keys.get("gemini")) and gemini_route not in tried and _route_health(*gemini_route)[0] <= time.monotonic()
        reserved = min(attempt_limit - 1, 1 if budget.get("syntax_corrections", 0) else 2)
        if provider != "Gemini" and gemini_pending and attempts >= attempt_limit - reserved:
            continue
        if provider == "OpenRouter" and not _free_available(model):
            logger.info("Coder skipping unavailable free route %s; recheck in 60 seconds", model)
            continue
        messages, correcting, quota_retried = base_messages, False, False
        while attempts < attempt_limit and time.monotonic() < deadline:
            input_tokens = _input_tokens(messages)
            cap = 16384
            if provider == "Groq":
                limit = min(7500, max(2048, int(os.getenv("CODER_GROQ_REQUEST_TOKEN_LIMIT", "7500"))))
                cap = min(4096, limit - input_tokens)
                if cap < 2048:
                    logger.info("Coder skipping oversized Groq request for %s; full requirements use fallback", model)
                    break
            until, waitable = _route_health(*route)
            with _state_lock:
                quota = _quotas.get(state_key)
            if quota and quota[0] < input_tokens + cap and quota[1] > time.monotonic():
                until, waitable = max(until, quota[1]), True
            delay = max(0, until - time.monotonic())
            if delay:
                if not waitable or delay + .05 > wait_limit - waited or delay + .05 >= deadline - time.monotonic():
                    logger.info("Coder cooling down %s / %s for %.1fs; eligible automatically afterward", provider, model, delay)
                    if route not in rechecked and route not in tried and route not in deferred:
                        deferred[route] = until
                    break
                time.sleep(delay + .05)
                waited += delay + .05
                budget["waited"] = waited
            request_timeout = timeout
            if provider == "Groq":
                request_timeout = min(timeout, max(1, float(os.getenv("CODER_GROQ_TIMEOUT_SECONDS", "15"))))
            elif model == NEMOTRON_FREE:
                request_timeout = min(120, max(1, float(os.getenv("CODER_NVIDIA_TIMEOUT_SECONDS", "60"))))
            remaining = deadline - time.monotonic()
            if provider != "Gemini" and gemini_pending:
                reserve_time = timeout * (1 if budget.get("syntax_corrections", 0) else 2)
                remaining -= min(reserve_time, remaining / 2)
            if remaining <= 0:
                break
            attempts += 1
            budget["attempts"] = attempts
            if correcting:
                budget["syntax_corrections"] = 1
            tried.add(route)
            code = ""
            try:
                logger.info("Coder request %s/%s: %s / %s%s", attempts, attempt_limit, provider, model,
                            " (syntax correction)" if correcting else "")
                response = _request(provider=provider, model=model, key=key, messages=messages,
                                    max_tokens=cap, timeout=min(request_timeout, remaining))
                with _state_lock:
                    if provider == "Groq" and response.headers.get("x-ratelimit-remaining-tokens") is not None:
                        _quotas[state_key] = (float(response.headers["x-ratelimit-remaining-tokens"]),
                            time.monotonic() + _seconds(response.headers.get("x-ratelimit-reset-tokens")))
                code = re.sub(r"<think>.*?</think>", "", response.code or "", flags=re.DOTALL)
                code = re.sub(r"^```(?:python)?\s*|^```\s*$", "", code, flags=re.MULTILINE).strip()
                if not code or not ast.parse(code).body:
                    raise DraftResponseError("empty_implementation")
                with _state_lock:
                    _cooldowns.pop(state_key, None)
                logger.info("Coder draft validated: %s / %s", provider, model)
                return code + "\n"
            except SyntaxError as exc:
                # Never log exc.msg, exc.text or source: syntax errors can echo
                # private literals. Only fixed exception types and positions.
                logger.warning("Coder syntax rejected: %s / %s (%s) line=%s column=%s",
                               provider, model, type(exc).__name__, exc.lineno, exc.offset)
                if budget.get("syntax_corrections", 0) or correcting:
                    break
                messages = [*base_messages, {"role": "assistant", "content": code}, {"role": "user", "content":
                    f"The previous module failed Python syntax validation ({type(exc).__name__}) at line {exc.lineno}, column {exc.offset}. "
                    "Correct the syntax and return the COMPLETE Python module. Preserve all original requirements, public interfaces and unrelated behavior. "
                    "Return code only, without Markdown fences, explanations or placeholders."}]
                correcting = True
            except DraftResponseError as exc:
                logger.warning("Coder response rejected: %s / %s category=%s", provider, model, exc.category)
                break
            except Exception as exc:
                status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
                text = str(exc).lower()
                headers = getattr(getattr(exc, "response", None), "headers", {}) or {}
                hint = re.search(r"try again in ([\d.a-z]+)", text)
                delay = max(_seconds(headers.get("retry-after"), 0),
                            _seconds(headers.get("x-ratelimit-reset-tokens"), 0),
                            _seconds(hint.group(1), 0) if hint else 0)
                oversized = "request too large" in text or "expected output tokens exceed" in text
                waitable = status == 429 and not oversized
                delay = delay or (3600 if "tokens per day" in text else 60 if status == 429 else 30)
                if isinstance(exc, (TimeoutError, httpx.TimeoutException, APITimeoutError)):
                    delay, waitable = max(delay, 180), False
                if oversized or status in (400, 404):
                    delay, waitable = 300, False
                with _state_lock:
                    _cooldowns[state_key] = (time.monotonic() + delay, waitable)
                    if status in (401, 402, 403):
                        _cooldowns[(hash(key), provider, "*")] = (time.monotonic() + 300, False)
                    if provider == "OpenRouter" and status == 429:
                        body = getattr(exc, "body", {}) or {}
                        error = body.get("error", body) if isinstance(body, dict) else {}
                        metadata = error.get("metadata") if isinstance(error, dict) else None
                        metadata = metadata if isinstance(metadata, dict) else {}
                        if not metadata.get("provider_name") and not metadata.get("provider_code"):
                            _cooldowns[(hash(key), provider, "*")] = (time.monotonic() + delay, False)
                logger.warning("Coder failed: %s / %s (%s); temporary cooldown %.1fs", provider, model,
                               status or type(exc).__name__, delay)
                if correcting or quota_retried or not waitable or delay + .05 > wait_limit - waited:
                    break
                quota_retried = True
    raise DraftGenerationError("Draft providers are temporarily unavailable or the generation budget was reached. Completed drafts are retained; retry only missing modules after provider cooldowns.")
