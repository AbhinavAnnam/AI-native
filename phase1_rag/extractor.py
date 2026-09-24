import concurrent.futures
import io
import json
import logging
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional, Type, Union

import pypdf
from google import genai
from google.genai import types
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

EXHAUSTED_MODELS = set()
EXHAUSTED_LOCK = threading.Lock()


def reset_exhausted_models():
    """Resets the global blacklisted models set prior to starting an extraction pipeline."""
    global EXHAUSTED_MODELS, EXHAUSTED_LOCK
    with EXHAUSTED_LOCK:
        EXHAUSTED_MODELS.clear()


def clean_json_text(text: str) -> str:
    """
    Cleans raw text before JSON parsing:
    1. Strips markdown code fences (e.g. ```json ... ```).
    2. Strips malformed array numeric key prefixes (e.g., '[ 0: {' or ', 1: {').
    """
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()

    # Pre-parse syntax repair for pseudo-array key prefixes like '[ 0: {' or ', 1: {'
    text = re.sub(r'(\[\s*)\d+\s*:\s*', r'\1', text)
    text = re.sub(r',\s*\d+\s*:\s*', ', ', text)

    return text.strip()


def normalize_dict_lists(data: Any) -> Any:
    """
    Recursively transforms dictionary objects with numeric string keys ('0', '1', ...)
    into clean JSON/Python lists.
    """
    if isinstance(data, dict):
        keys = list(data.keys())
        if keys and all(k.isdigit() for k in keys):
            sorted_keys = sorted(keys, key=lambda k: int(k))
            return [normalize_dict_lists(data[k]) for k in sorted_keys]
        return {k: normalize_dict_lists(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [normalize_dict_lists(item) for item in data]
    return data


def call_gemini_with_fallback(
    client: genai.Client,
    models_in_priority: List[str],
    contents: str,
    config: types.GenerateContentConfig,
    max_retries_per_model: int = 2,
) -> Any:
    """Executes a Gemini API call with thread-safe global model blacklisting."""
    global EXHAUSTED_MODELS, EXHAUSTED_LOCK

    for model in models_in_priority:
        with EXHAUSTED_LOCK:
            if model in EXHAUSTED_MODELS:
                continue

        delay = 1.0
        for attempt in range(1, max_retries_per_model + 1):
            with EXHAUSTED_LOCK:
                if model in EXHAUSTED_MODELS:
                    logger.info(
                        f"Skipping {model} (Attempt {attempt}) - blacklisted by parallel worker."
                    )
                    break

            try:
                logger.info(f"API Request -> Model: {model} (Attempt {attempt})")
                return client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )
            except Exception as e:
                err_msg = str(e)
                err_code = getattr(e, "code", None)

                is_quota_exceeded = (
                    "429" in err_msg
                    or "RESOURCE_EXHAUSTED" in err_msg
                    or err_code == 429
                )
                is_transient = (
                    err_code in [500, 502, 503, 504]
                    or any(code in err_msg for code in ["503", "500", "502", "504"])
                    or any(
                        term in err_msg
                        for term in [
                            "UNAVAILABLE",
                            "high demand",
                            "TEMPORARY",
                            "overloaded",
                        ]
                    )
                )

                if is_quota_exceeded:
                    logger.warning(
                        f"Model {model} hit 429 Rate/Quota limit. Blacklisting globally across all threads."
                    )
                    with EXHAUSTED_LOCK:
                        EXHAUSTED_MODELS.add(model)
                    break

                elif is_transient:
                    logger.warning(
                        f"Model {model} hit transient error (Attempt {attempt}/{max_retries_per_model}): {err_msg}"
                    )
                    if attempt < max_retries_per_model:
                        time.sleep(delay)
                        delay *= 2.0
                    else:
                        logger.warning(
                            f"Exhausted retries for {model}. Blacklisting globally across all threads..."
                        )
                        with EXHAUSTED_LOCK:
                            EXHAUSTED_MODELS.add(model)
                        break
                else:
                    raise e

    with EXHAUSTED_LOCK:
        blacklisted = list(EXHAUSTED_MODELS)
    raise RuntimeError(
        f"All candidate models exhausted or failed: {models_in_priority}. Blacklisted set: {blacklisted}"
    )


def extract_raw_text_from_file_object(file_obj: Any) -> str:
    """Extracts plain text from PDF files or stream objects."""
    content_bytes = None

    if hasattr(file_obj, "file"):
        file_obj.file.seek(0)
        content_bytes = file_obj.file.read()
    elif hasattr(file_obj, "read"):
        if callable(file_obj.read):
            content_bytes = file_obj.read()
    elif isinstance(file_obj, bytes):
        content_bytes = file_obj
    elif isinstance(file_obj, str) and os.path.exists(file_obj):
        reader = pypdf.PdfReader(file_obj)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages)

    if not content_bytes:
        raise ValueError("Could not read binary content from uploaded file.")

    reader = pypdf.PdfReader(io.BytesIO(content_bytes))
    extracted_pages = []
    for idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            extracted_pages.append(f"--- PAGE {idx + 1} ---\n{text}")

    return "\n\n".join(extracted_pages).strip()


def chunk_text(text: str, chunk_size: int = 12000, overlap: int = 1500) -> List[str]:
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += chunk_size - overlap
    return chunks


def index_chunks_in_chroma(
    chunks: List[str], embedder: Any, chroma_client: Any, collection_name: str = "phase1_specs"
):
    if not chroma_client or not embedder:
        return

    try:
        logger.info(f"Indexing {len(chunks)} chunk(s) into ChromaDB collection '{collection_name}'...")
        collection = chroma_client.get_or_create_collection(name=collection_name)
        
        embeddings = embedder.encode(chunks).tolist() if hasattr(embedder, "encode") else None
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        
        if embeddings:
            collection.upsert(ids=ids, documents=chunks, embeddings=embeddings)
        else:
            collection.upsert(ids=ids, documents=chunks)
            
        logger.info("ChromaDB indexing complete.")
    except Exception as e:
        logger.warning(f"Failed to index chunks into ChromaDB: {e}")


def map_chunk_worker(
    idx: int,
    total_chunks: int,
    chunk: str,
    client: genai.Client,
    model_candidates: List[str],
) -> Optional[Dict[str, Any]]:
    logger.info(f"Processing chunk {idx}/{total_chunks}...")

    prompt = f"""You are a senior technical specification extraction system.
Extract all structural software architecture elements from this document chunk into a JSON structure.

Include:
- Functional modules/services and snake_case target module names derived directly from the text.
- Business rules, domain policies, validation logic, and edge cases.
- Function signatures with explicit parameter names, types, and return types.
- System, domain, and validation exceptions/errors.
- Cross-cutting constraints and technical guardrails (e.g. security, tenant isolation, path traversal, rate limits, concurrency, time/timezones, data retention, error recovery, background jobs).

Output MUST be standard JSON. Do NOT prefix array elements with numeric keys (e.g., do NOT write `0: {{}}`).

Document Chunk:
{chunk}
"""

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=0.1,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(
            disable=True
        ),
    )

    try:
        response = call_gemini_with_fallback(
            client=client,
            models_in_priority=model_candidates,
            contents=prompt,
            config=config,
        )
        if response and response.text:
            cleaned = clean_json_text(response.text)
            parsed = json.loads(cleaned)
            return normalize_dict_lists(parsed)
    except json.JSONDecodeError as err:
        logger.warning(f"Failed to parse JSON response for chunk {idx}: {err}")
    except Exception as e:
        logger.error(f"Error processing chunk {idx}: {e}")
    return None


def run_extraction_pipeline(
    files: Any,
    gemini_key: str,
    embedder: Optional[Any] = None,
    chroma_client: Optional[Any] = None,
    response_schema: Optional[Type[BaseModel]] = None,
    model_priority: Optional[List[str]] = None,
    max_parallel_workers: int = 4,
    **kwargs: Any,
) -> Dict[str, Any]:
    reset_exhausted_models()

    if not gemini_key:
        raise ValueError("Missing GEMINI_API_KEY.")

    if not model_priority:
        model_priority = [
            "gemini-3.6-flash",
            "gemini-3.5-flash-lite",
            "gemini-3.5-flash",
        ]

    client = genai.Client(api_key=gemini_key)

    file_list = files if isinstance(files, list) else [files]
    extracted_texts = []
    for f in file_list:
        text = extract_raw_text_from_file_object(f)
        if text.strip():
            extracted_texts.append(text)

    raw_text = "\n\n".join(extracted_texts).strip()
    if not raw_text:
        raise ValueError("Unable to extract text from uploaded document(s).")

    chunks = chunk_text(raw_text, chunk_size=12000, overlap=1500)
    logger.info(f"Document chunked into {len(chunks)} segment(s).")

    if embedder and chroma_client:
        index_chunks_in_chroma(chunks, embedder, chroma_client)

    intermediate_results = {}
    if len(chunks) == 1:
        res = map_chunk_worker(1, 1, chunks[0], client, model_priority)
        if res:
            intermediate_results[1] = res
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel_workers) as executor:
            future_to_idx = {
                executor.submit(
                    map_chunk_worker, idx, len(chunks), chunk, client, model_priority
                ): idx
                for idx, chunk in enumerate(chunks, start=1)
            }

            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                res = future.result()
                if res:
                    intermediate_results[idx] = res

    ordered_extractions = [
        intermediate_results[i] for i in sorted(intermediate_results.keys())
    ]

    logger.info("Reduce Phase: Consolidating extractions into final specification schema...")

    reduce_prompt = f"""You are a principal software architect. Consolidate these chunk-level extractions into a complete, unified software specification schema in JSON format matching the response schema.

CONSOLIDATION & STRUCTURAL REQUIREMENTS:
1. "features" MUST be a standard JSON array of feature objects (`[...]`). Do NOT use key-indexed objects (e.g. do not use "0", "1") or array prefixes like `0: {{}}`.
2. DYNAMIC MODULE CONSOLIDATION (PREVENT OVER-FRAGMENTATION):
   - Do NOT create tiny, granular micro-modules (e.g. avoid separate modules for comments, invitations, password reset, or dashboard).
   - Dynamically aggregate all related capabilities into AT MOST 3 to 5 core macro domain contexts based on shared entity boundaries and relationships (e.g. aggregate identity/accounts/invitations into one identity context; projects/tasks/comments into one work execution context).
   - Dynamically assign a clean, snake_case `target_module` name for each macro domain context.
   - Deduplicate and merge all business rules, function signatures, and exception classes within each consolidated macro-module.
3. Extract all cross-cutting, architectural, security, performance, and operational constraints into the `cross_cutting_constraints` list. Each entry must have a snake_case `category_name` (e.g., `security_and_authorization`, `concurrency_and_locking`, `temporal_and_timezones`, `background_processing_and_retries`, `data_retention`) and a list of explicit `rules`.
4. Every rule entry under `cross_cutting_constraints` must contain explicit, actionable text extracted from the document. Do NOT output generic placeholder functions or empty signatures.

Chunk Extractions:
{json.dumps(ordered_extractions, indent=2)}
"""

    reduce_config = types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=0.1,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(
            disable=True
        ),
    )
    if response_schema:
        reduce_config.response_schema = response_schema

    reduce_response = call_gemini_with_fallback(
        client=client,
        models_in_priority=model_priority,
        contents=reduce_prompt,
        config=reduce_config,
    )

    raw_data = None
    if hasattr(reduce_response, "parsed") and reduce_response.parsed:
        parsed_data = reduce_response.parsed
        if hasattr(parsed_data, "model_dump"):
            raw_data = parsed_data.model_dump()
        elif hasattr(parsed_data, "dict"):
            raw_data = parsed_data.dict()

    if not raw_data and reduce_response and reduce_response.text:
        cleaned = clean_json_text(reduce_response.text)
        raw_data = json.loads(cleaned)

    if raw_data:
        normalized_data = normalize_dict_lists(raw_data)
        if response_schema:
            try:
                validated_model = response_schema.model_validate(normalized_data)
                return validated_model.model_dump()
            except Exception as val_err:
                logger.warning(f"Pydantic schema re-validation notice: {val_err}")
                return normalized_data
        return normalized_data

    raise RuntimeError("Failed to obtain valid specification output from Gemini.")


extract_document_spec = run_extraction_pipeline