import logging
import os
import time
from typing import List, Literal

from langchain_google_genai import ChatGoogleGenerativeAI

from backend.models.schemas import TaskClassification

logger = logging.getLogger(__name__)

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
        delay = 1.0
        for model in self.candidates:
            for attempt in range(2):
                try:
                    llm = ChatGoogleGenerativeAI(
                        model=model,
                        google_api_key=self.api_key,
                        temperature=self.temperature,
                    )
                    return llm.invoke(messages)
                except Exception as exc:
                    last_err = exc
                    logger.warning(
                        f"[LLMRouter] code model '{model}' failed "
                        f"(attempt {attempt + 1}/2): {exc}"
                    )
                    if _is_transient(exc):
                        time.sleep(delay)
                        delay *= 2.0
                        continue
                    break
        raise last_err


class LLMRouter:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY") or ""

    def _run_with_fallback(self, candidates, temperature, prompt, response_schema=None):
        last_err: Exception = None
        delay = 1.0
        for model in candidates:
            for attempt in range(2):
                try:
                    llm = ChatGoogleGenerativeAI(
                        model=model,
                        google_api_key=self.api_key,
                        temperature=temperature,
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
                    if _is_transient(exc):
                        time.sleep(delay)
                        delay *= 2.0
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
