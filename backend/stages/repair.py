"""Bounded text generation for critic repairs using the existing provider keys."""
import asyncio
import logging
import os
import ast
import re
import time

from google import genai
from google.genai import types
from phase2_coder.providers import FREE_CODER, _free_available

from phase1_rag.providers import (_critic_request, critic_request_timeout, ProviderBillingError,
    ProviderRateLimitError, ReviewTimeLimitError, critic_groq_limit, critic_token_estimate,
    critic_openrouter_model, critic_gemini_models, critic_provider_ready, remember_critic_failure,
    check_critic_quota, remember_critic_quota, ProviderUnavailableError)

logger = logging.getLogger(__name__)


def _gemini_repair(key, model, prompt):
    timeout = critic_request_timeout()
    async def request():
        async with genai.Client(api_key=key, http_options=types.HttpOptions(
            timeout=int(timeout * 1000), retry_options=types.HttpRetryOptions(attempts=1))).aio as client:
            response = await client.models.generate_content(model=model, contents=prompt,
                config=types.GenerateContentConfig(max_output_tokens=16384,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)))
            if not response.candidates or str(response.candidates[0].finish_reason).split(".")[-1] != "STOP":
                raise ValueError("Truncated repair")
            return response.text
    return asyncio.run(asyncio.wait_for(request(), timeout=timeout))


def generate_repair_text(prompt, system_prompt, *, max_attempts=3, keys=None):
    keys = keys or {}
    groq = keys.get("groq") or os.getenv("GROQ_API_KEY") or os.getenv("groq_api_key")
    router = keys.get("openrouter") or os.getenv("OPENROUTER_API_KEY") or os.getenv("openrouter_api_key")
    gemini = keys.get("gemini") or os.getenv("GEMINI_API_KEY")
    candidates = []
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
    available = critic_groq_limit() - critic_token_estimate(messages, 0)
    groq_output = min(4096, available)
    # The same completion cap is checked here AND sent to the provider.
    if groq and groq_output >= 2048:
        candidates.append(("Groq", "openai/gpt-oss-20b", groq, "https://api.groq.com/openai/v1"))
    if router:
        preferred = critic_openrouter_model()
        free_models = ([preferred] if preferred != "openrouter/free" else []) + [FREE_CODER, "openrouter/free"]
        for model in dict.fromkeys(free_models):
            candidates.append(("OpenRouter", model, router, "https://openrouter.ai/api/v1"))
    if gemini:
        candidates.extend(("Gemini", model, gemini, None) for model in critic_gemini_models())
    billing_error = None
    quota_error = None
    transient_error = False
    attempts, google_blocked, router_blocked = 0, False, False
    retried = set()
    # A queue allows one short server-error retry after independent alternatives.
    candidates = [(*candidate, False) for candidate in candidates]
    for provider, model, key, url, retry in candidates:
        if attempts >= max_attempts:
            break
        if (provider == "OpenRouter" and (billing_error or router_blocked)) or (provider == "Gemini" and google_blocked):
            continue
        if not retry and not critic_provider_ready(key, provider, model):
            logger.info("Critic repair skipping cooling provider: %s / %s", provider, model)
            continue
        # Only consume inference attempts on available, verified free routes.
        if provider == "OpenRouter" and model == FREE_CODER and not _free_available(model):
            continue
        # Keep one slot for an independent Google fallback if configured.
        if provider != "Gemini" and gemini and attempts == max_attempts - 1 and not google_blocked:
            continue
        critic_request_timeout()
        try:
            check_critic_quota(key, provider, model)
            if retry:
                if critic_request_timeout() <= 1:
                    raise ReviewTimeLimitError("Repair time budget reached. Completed files are saved.")
                time.sleep(1)
            attempts += 1
            logger.info("Critic repair request %s/%s: %s / %s", attempts, max_attempts, provider, model)
            if provider == "Gemini":
                code = _gemini_repair(key, model, system_prompt + "\n\n" + prompt)
            else:
                extra = {"reasoning": {"exclude": True}} if provider == "OpenRouter" else {"include_reasoning": False}
                kwargs = {} if provider == "OpenRouter" else {"reasoning_effort": "low"}
                raw = _critic_request(key, url, model=model,
                    messages=messages,
                    temperature=.1, max_tokens=16384 if provider == "OpenRouter" else groq_output,
                    extra_body=extra, **kwargs)
                choice = raw.parse().choices[0]
                if choice.finish_reason != "stop":
                    raise ValueError("Incomplete repair")
                code = choice.message.content
            if not code or not code.strip():
                raise ValueError("Empty repair")
            clean = re.sub(r"^```(?:python)?\s*|^```\s*$", "", code, flags=re.MULTILINE).strip()
            if not ast.parse(clean).body:
                raise ValueError("Empty implementation")
            from phase1_rag.providers import _critic_failures
            _critic_failures.pop((hash(key), provider, model), None)
            return clean + "\n"
        except ReviewTimeLimitError:
            raise
        except ProviderRateLimitError as exc:
            quota_error = exc
            google_blocked |= provider == "Gemini"
            router_blocked |= provider == "OpenRouter"
        except Exception as exc:
            remember_critic_failure(key, provider, model, exc)
            status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
            logger.warning("Critic repair failed: %s / %s (%s) status=%s", provider, model, type(exc).__name__, status or "unknown")
            if status == 402:
                billing_error = ProviderBillingError(provider)
            if status == 429 or type(exc).__name__ == "GoogleRateLimitError":
                quota_error = remember_critic_quota(key, provider, model, exc)
                google_blocked |= provider == "Gemini"
                router_blocked |= provider == "OpenRouter"
            if status in {401, 403}:
                google_blocked |= provider == "Gemini"
                router_blocked |= provider == "OpenRouter"
            if isinstance(exc, TimeoutError) or type(exc).__name__ in {"APITimeoutError", "ReadTimeout", "APIConnectionError", "ConnectError"}:
                transient_error = True
            if (isinstance(status, int) and 500 <= status <= 599) or type(exc).__name__ == "ServerError":
                transient_error = True
                identity = (provider, model)
                if identity not in retried:
                    retried.add(identity)
                    candidates.append((provider, model, key, url, True))
    if billing_error:
        raise billing_error
    if quota_error:
        raise quota_error
    if transient_error or (candidates and attempts == 0):
        raise ProviderUnavailableError()
    raise RuntimeError("Repair providers could not return complete code. Retry when a provider is available.")
