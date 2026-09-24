import io
import json
import logging
import time
from typing import Any, Dict, List, Optional, Type

import pypdf
from google import genai
from google.genai import types
from google.genai.errors import APIError
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Track exhausted models for the execution session to prevent repeated rate-limit loops
EXHAUSTED_MODELS = set()


# =====================================================================
# 1. PYDANTIC SCHEMAS FOR CODER AGENT FIDELITY
# =====================================================================

class FunctionSpec(BaseModel):
    signature: str = Field(description="Normalized pythonic function signature with typed parameters and return type.")
    purpose: str = Field(description="Exact operational purpose of the function.")


class FeatureModule(BaseModel):
    feature_name: str = Field(description="Human-readable module name.")
    target_module: str = Field(description="Snake_case module identifier.")
    business_rules: List[str] = Field(description="Functional domain rules, state transition rules, and validation requirements.")
    functions: List[FunctionSpec] = Field(description="Explicit API or internal function signatures needed for this module.")
    exceptions: List[str] = Field(description="Error handling, validation failures, and edge cases specific to this module.")


class CrossCuttingRules(BaseModel):
    concurrency_and_race_conditions: List[str] = Field(description="Rules covering concurrent edits, idempotency, race condition prevention, and double-submit guards.")
    temporal_and_timezone_mechanics: List[str] = Field(description="Rules for UTC storage, display timezones, DST handling, calendar vs instant semantics, and relative dates.")
    background_processing_and_retries: List[str] = Field(description="Asynchronous job requirements, retry strategies, idempotency keys, and side-effect isolation.")
    security_and_tenant_isolation: List[str] = Field(description="Multi-tenancy isolation rules, authorization, file storage security, path traversal prevention, and token safety.")
    data_retention_and_historical_integrity: List[str] = Field(description="Soft deletion, deactivation semantics, append-only logs, and historical display name preservation.")


class SystemSpecification(BaseModel):
    project_title: str = Field(description="Name of the project or workspace.")
    features: List[FeatureModule] = Field(description="Bounded domain modules containing core workflows and logic.")
    cross_cutting_constraints: CrossCuttingRules = Field(description="System-wide non-functional requirements essential for implementation correctness.")


# =====================================================================
# 2. MAP & REDUCE PROMPTS (ENFORCING ZERO-DROP)
# =====================================================================

MAP_SYSTEM_PROMPT = """
You are an expert Systems Architect. Analyze this contiguous section of project documentation and extract EVERY single detail, rule, requirement, edge case, and architectural constraint.

EXTRACT EVERYTHING IN THE FOLLOWING BUCKETS:
1. FUNCTIONAL LOGIC: Function ideas, arguments, state transitions, validation rules, business logic.
2. CONCURRENCY & RACES: Idempotency, simultaneous edits, duplicate request handling, retry protection.
3. TEMPORAL/TIMEZONE RULES: UTC conversion, calendar vs instant semantics, relative dates, DST logic.
4. BACKGROUND & RETRIES: Asynchronous processing, failed mail/indexing isolation, background job safety.
5. SECURITY & TENANT ISOLATION: Cross-tenant leakage guards, token semantics, file download authorization, path traversal rules.
6. DATA INTEGRITY: Account deactivation history preservation, soft deletes, activity audit trails.

Do NOT summarize. Extract raw exact logic regardless of formatting or prose style.
"""

REDUCE_SYSTEM_PROMPT = """
You are a Principal Software Architect building a specification for an automated Coder Agent.
You are given raw extractions from an entire documentation set.

CONSOLIDATION & ZERO-DROP RULES:
1. Preserve Every Invariant: Your output MUST capture every business rule, edge case, concurrency guarantee, background retry behavior, and timezone rule mentioned.
2. Map to Dual Hierarchy:
   - Functional capabilities go into `features` (bounded domain modules).
   - Non-functional, cross-cutting rules (concurrency, background retries, DST, tenant isolation, historical data preservation) MUST go into `cross_cutting_constraints`.
3. Standardize Signatures: All functions inside `features` must have snake_case identifiers, typed arguments, and explicit return signatures.
4. Zero Invention: Do not invent rules not mentioned, but strictly preserve 100% of stated logic.
"""


# =====================================================================
# 3. UTILITIES & RETRY LOGIC WITH CIRCUIT BREAKER
# =====================================================================

def _extract_text_from_uploads(files: List[Any]) -> str:
    combined_text = []
    for file in files:
        filename = getattr(file, "filename", "uploaded_doc")
        logger.info(f"Parsing uploaded file: {filename}")
        file_bytes = file.file.read()

        if filename.lower().endswith(".pdf"):
            pdf_reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            for page_idx, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text:
                    combined_text.append(
                        f"--- Document: {filename} (Page {page_idx + 1}) ---\n{text}"
                    )
        else:
            text_content = file_bytes.decode("utf-8", errors="ignore")
            combined_text.append(f"--- Document: {filename} ---\n{text_content}")

    return "\n\n".join(combined_text)


def _chunk_text_contiguous(
    text: str, chunk_size: int = 12000, overlap: int = 1500
) -> List[str]:
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def _call_gemini_with_fallback(
    client: genai.Client,
    models_in_priority: List[str],
    contents: str,
    config: types.GenerateContentConfig,
    max_retries_per_model: int = 2,
) -> Any:
    """Tries primary model first. If quota/server overload is hit, permanently flags the model 
    as exhausted for the run and moves to the next candidate model."""
    global EXHAUSTED_MODELS

    # Filter out models that already hit hard rate limits/quota/server errors in previous calls
    active_models = [m for m in models_in_priority if m not in EXHAUSTED_MODELS]

    if not active_models:
        raise RuntimeError(
            f"All candidate models are exhausted for this session: {models_in_priority}"
        )

    for model in active_models:
        delay = 2.0
        for attempt in range(1, max_retries_per_model + 1):
            try:
                logger.info(
                    f"Attempting API call with model: {model} (Attempt {attempt})..."
                )
                return client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )
            except Exception as e:
                err_msg = str(e)
                err_code = getattr(e, "code", None)

                # Catch rate limits (429), server overload (503), temporary outages (500, 502, 504), and status string keywords
                is_transient_or_rate_limit = (
                    err_code in [429, 500, 502, 503, 504]
                    or any(code in err_msg for code in ["429", "503", "500", "502", "504"])
                    or any(
                        term in err_msg
                        for term in [
                            "RESOURCE_EXHAUSTED",
                            "UNAVAILABLE",
                            "high demand",
                            "TEMPORARY",
                            "overloaded",
                        ]
                    )
                )

                if is_transient_or_rate_limit:
                    logger.warning(
                        f"Model {model} hit transient error/rate-limit (Attempt {attempt}/{max_retries_per_model}): {err_msg}"
                    )
                    if attempt < max_retries_per_model:
                        time.sleep(delay)
                        delay *= 2.0
                    else:
                        logger.warning(
                            f"Exhausted retries for {model}. Marking as exhausted for session and falling back..."
                        )
                        EXHAUSTED_MODELS.add(model)
                else:
                    # Non-retryable error (e.g. invalid arguments); raise immediately
                    raise e

    raise RuntimeError(
        f"All requested models failed due to quota, capacity limits, or server errors: {models_in_priority}"
    )


# =====================================================================
# 4. PIPELINE PHASES
# =====================================================================

def _map_phase(
    chunks: List[str], client: genai.Client, model_candidates: List[str]
) -> List[Dict[str, Any]]:
    intermediate_extractions = []

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=0.1,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(
            disable=True
        ),
    )

    for idx, chunk in enumerate(chunks, start=1):
        logger.info(f"Map Phase: Processing chunk {idx}/{len(chunks)}...")
        prompt = f"{MAP_SYSTEM_PROMPT}\n\nText Segment ({idx}/{len(chunks)}):\n{chunk}"

        response = _call_gemini_with_fallback(
            client=client,
            models_in_priority=model_candidates,
            contents=prompt,
            config=config,
        )

        if response and response.text:
            try:
                data = json.loads(response.text)
                intermediate_extractions.append(data)
            except json.JSONDecodeError:
                logger.warning(
                    f"Failed to parse JSON for chunk {idx}. Continuing..."
                )

        time.sleep(1.0)

    return intermediate_extractions


def _reduce_phase(
    intermediate_data: List[Dict[str, Any]],
    client: genai.Client,
    model_candidates: List[str],
    response_schema: Type[BaseModel],
) -> Dict[str, Any]:
    logger.info(
        "Reduce Phase: Consolidating extractions into final specification schema..."
    )

    prompt = f"""
{REDUCE_SYSTEM_PROMPT}

Collected Raw System Extractions:
{json.dumps(intermediate_data)}

Task:
Consolidate all functional features and cross-cutting architectural constraints strictly according to the response schema.
"""

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=response_schema,
        temperature=0.1,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(
            disable=True
        ),
    )

    response = _call_gemini_with_fallback(
        client=client,
        models_in_priority=model_candidates,
        contents=prompt,
        config=config,
    )

    if response and response.text:
        return json.loads(response.text)

    raise RuntimeError("Reduce phase failed to output valid schema JSON.")


# =====================================================================
# 5. MAIN ENTRY POINT
# =====================================================================

def run_extraction_pipeline(
    files: List[Any],
    gemini_key: str,
    embedder: Any = None,
    chroma_client: Any = None,
    response_schema: Optional[Type[BaseModel]] = None,
) -> Dict[str, Any]:

    global EXHAUSTED_MODELS
    EXHAUSTED_MODELS.clear()  # Reset Circuit Breaker for each new run

    target_schema = response_schema or SystemSpecification

    raw_text = _extract_text_from_uploads(files)
    if not raw_text.strip():
        raise ValueError("No text could be extracted from the uploaded files.")

    chunks = _chunk_text_contiguous(raw_text)
    logger.info(
        f"Split raw document ({len(raw_text)} chars) into {len(chunks)} contiguous chunks."
    )

    client = genai.Client(api_key=gemini_key)

    # Candidate models in order of priority
    model_candidates = [
        "gemini-3.6-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.5-flash",
    ]

    raw_extractions = _map_phase(
        chunks, client, model_candidates=model_candidates
    )

    final_spec = _reduce_phase(
        raw_extractions,
        client,
        model_candidates=model_candidates,
        response_schema=target_schema,
    )

    return final_spec