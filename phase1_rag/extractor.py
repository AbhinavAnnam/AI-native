import concurrent.futures
import io
import json
import logging
import os
import re
import threading
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Type, Union

import pypdf
from google import genai
from google.genai import types
from pydantic import BaseModel
from phase1_rag.providers import fallback_keys_available, generate_json

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Set for permanently broken/non-existent models during a single execution run
EXHAUSTED_MODELS = set()
EXHAUSTED_LOCK = threading.Lock()


def reset_exhausted_models():
    """Resets permanent model blacklists before extraction."""
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


def minify_extractions(data: Any) -> Any:
    """
    Strips empty strings, nulls, and empty lists/dicts to minimize prompt payload size.
    """
    if isinstance(data, dict):
        cleaned_dict = {
            k: minify_extractions(v)
            for k, v in data.items()
            if v not in (None, "", [], {})
        }
        return {k: v for k, v in cleaned_dict.items() if v not in (None, "", [], {})}
    elif isinstance(data, list):
        cleaned_list = [
            minify_extractions(item)
            for item in data
            if item not in (None, "", [], {})
        ]
        return [item for item in cleaned_list if item not in (None, "", [], {})]
    return data


def prune_payload_for_fallback(data: Any, max_chars: int = 12000) -> str:
    """
    Aggressively minifies and prunes intermediate extractions to ensure
    the Reduce prompt payload fits within Groq's strict token/character limits (prevents 413 errors).
    """
    minified = minify_extractions(data)
    json_str = json.dumps(minified, separators=(",", ":"))

    if len(json_str) <= max_chars:
        return json_str

    # If payload is still too large for Groq, trim deeper redundant keys
    logger.warning(f"Payload size ({len(json_str)} chars) exceeds Groq limit ({max_chars}). Compacting payload...")
    
    if isinstance(minified, list):
        pruned_list = []
        for item in minified:
            if isinstance(item, dict):
                # Keep core architectural keys, drop verbose explanations
                pruned_item = {
                    k: v for k, v in item.items()
                    if k in ("features", "modules", "services", "functions", "cross_cutting_constraints")
                }
                pruned_list.append(pruned_item)
            else:
                pruned_list.append(item)
        return json.dumps(pruned_list, separators=(",", ":"))

    return json_str


def generate_json_with_retry(contents: str, response_schema: Any = None, max_retries: int = 2) -> Any:
    """
    Calls generate_json with backoff retry to handle transient Groq 429 Rate Limit errors
    without triggering long OpenRouter fallbacks.
    """
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            return generate_json(contents, response_schema=response_schema)
        except Exception as e:
            last_err = e
            err_msg = str(e)
            if "429" in err_msg and attempt < max_retries:
                wait_time = (attempt + 1) * 1.5
                logger.warning(f"Groq Rate Limit (429) hit. Retrying in {wait_time}s (Attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
            else:
                raise e
    raise last_err


def call_gemini_with_fallback(
    client: genai.Client,
    models_in_priority: List[str],
    contents: str,
    config: types.GenerateContentConfig,
) -> Any:
    """
    Instant Failover Execution for Gemini Models.
    Tries each candidate model once, instantly switching on failure.
    """
    global EXHAUSTED_MODELS, EXHAUSTED_LOCK

    last_exception = None

    for raw_model in models_in_priority:
        model = raw_model.replace("models/", "").strip()

        with EXHAUSTED_LOCK:
            if model in EXHAUSTED_MODELS:
                continue

        try:
            logger.info(f"API Request -> Gemini Model: {model}")
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            return response

        except Exception as e:
            last_exception = e
            err_msg = str(e)
            err_code = getattr(e, "code", None)

            is_not_found = err_code == 404 or "404" in err_msg or "Not Found" in err_msg
            if is_not_found:
                logger.warning(f"Model {model} returned 404. Blacklisting for this run.")
                with EXHAUSTED_LOCK:
                    EXHAUSTED_MODELS.add(model)
            else:
                logger.warning(f"Gemini Model {model} failed ({err_code or err_msg}). Switching to next Gemini model...")

    raise RuntimeError(f"All Gemini models exhausted. Last error: {last_exception}")


def call_extraction_with_fallback(
    client: Optional[genai.Client],
    models_in_priority: List[str],
    contents: str,
    config: types.GenerateContentConfig,
) -> Any:
    """
    PRIMARY: Gemini Models
    FALLBACK: Groq / OpenRouter (with 429 retry handling)
    """
    gemini_error = "Gemini client not initialized."

    if client is not None:
        try:
            return call_gemini_with_fallback(client, models_in_priority, contents, config)
        except Exception as exc:
            gemini_error = str(exc)
            logger.warning("Gemini execution failed: %s. Rotating to Groq/OpenRouter fallback...", gemini_error)

    if fallback_keys_available():
        try:
            logger.info("API Request -> Provider Fallback (Groq/OpenRouter)")
            data = generate_json_with_retry(contents, response_schema=config.response_schema)
            return SimpleNamespace(text=json.dumps(data), parsed=None)
        except Exception as exc:
            fallback_error = str(exc)
            logger.error("Groq/OpenRouter fallback failed: %s", fallback_error)
            raise RuntimeError(
                f"Both Gemini and Fallback providers failed. Gemini: {gemini_error} | Fallback: {fallback_error}"
            )

    raise RuntimeError(f"Gemini failed ({gemini_error}) and no Groq/OpenRouter fallback keys available.")


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


def chunk_text(text: str, chunk_size: int = 25000, overlap: int = 1500) -> List[str]:
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
    client: Optional[genai.Client],
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
        response = call_extraction_with_fallback(
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
    gemini_key: str = "",
    embedder: Optional[Any] = None,
    chroma_client: Optional[Any] = None,
    response_schema: Optional[Type[BaseModel]] = None,
    model_priority: Optional[List[str]] = None,
    max_parallel_workers: int = 5,
    chunk_size: int = 25000,
    overlap: int = 1500,
    **kwargs: Any,
) -> Dict[str, Any]:
    reset_exhausted_models()

    if not gemini_key and not fallback_keys_available():
        raise ValueError("Set GROQ_API_KEY, OPENROUTER_API_KEY, or GEMINI_API_KEY.")

    if not model_priority:
        # Original Gemini model list
        model_priority = [
            "gemini-3.6-flash",
            "gemini-3.5-flash-lite",
            "gemini-3.5-flash",
            "gemini-3.1-flash-lite",
            "gemini-2.5-flash-lite",
        ]

    client = genai.Client(api_key=gemini_key) if gemini_key else None

    file_list = files if isinstance(files, list) else [files]
    extracted_texts = []
    for f in file_list:
        text = extract_raw_text_from_file_object(f)
        if text.strip():
            extracted_texts.append(text)

    raw_text = "\n\n".join(extracted_texts).strip()
    if not raw_text:
        raise ValueError("Unable to extract text from uploaded document(s).")

    chunks = chunk_text(raw_text, chunk_size=chunk_size, overlap=overlap)
    logger.info(f"Document chunked into {len(chunks)} segment(s).")

    if embedder and chroma_client:
        index_chunks_in_chroma(chunks, embedder, chroma_client)

    intermediate_results = {}
    if len(chunks) == 1:
        res = map_chunk_worker(1, 1, chunks[0], client, model_priority)
        if res:
            intermediate_results[1] = res
    else:
        try:
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
        except KeyboardInterrupt:
            logger.warning("Pipeline execution interrupted by user (Ctrl+C). Stopping workers...")
            executor.shutdown(wait=False, cancel_futures=True)
            raise

    if len(intermediate_results) != len(chunks):
        missing = [str(i) for i in range(1, len(chunks) + 1) if i not in intermediate_results]
        raise RuntimeError(
            "Extraction incomplete. Failed chunks: " + ", ".join(missing)
            + ". Check provider errors in the log."
        )

    ordered_extractions = [
        intermediate_results[i] for i in sorted(intermediate_results.keys())
    ]

    logger.info("Reduce Phase: Consolidating extractions into final specification schema...")

    # Compact & prune intermediate payload specifically for fallback/Groq context size rules
    compact_json_str = prune_payload_for_fallback(ordered_extractions)

    reduce_prompt = f"""You are a principal software architect. Consolidate these chunk-level extractions into a complete, unified software specification schema in JSON format matching the response schema.

CONSOLIDATION & STRUCTURAL REQUIREMENTS:
1. "features" MUST be a standard JSON array of feature objects (`[...]`). Do NOT use key-indexed objects (e.g. do NOT use "0", "1") or array prefixes like `0: {{}}`.
2. DYNAMIC MODULE CONSOLIDATION (PREVENT OVER-FRAGMENTATION):
   - Do NOT create tiny, granular micro-modules (e.g. avoid separate modules for comments, invitations, password reset, or dashboard).
   - Dynamically aggregate all related capabilities into AT MOST 3 to 5 core macro domain contexts based on shared entity boundaries and relationships (e.g. aggregate identity/accounts/invitations into one identity context; projects/tasks/comments into one work execution context).
   - Dynamically assign a clean, snake_case `target_module` name for each macro domain context.
   - Deduplicate and merge all business rules, function signatures, and exception classes within each consolidated macro-module.
3. Extract all cross-cutting, architectural, security, performance, and operational constraints into the `cross_cutting_constraints` list. Each entry must have a snake_case `category_name` (e.g., `security_and_authorization`, `concurrency_and_locking`, `temporal_and_timezones`, `background_processing_and_retries`, `data_retention`) and a list of explicit `rules`.
4. Every rule entry under `cross_cutting_constraints` must contain explicit, actionable text extracted from the document. Do NOT output generic placeholder functions or empty signatures.

Chunk Extractions:
{compact_json_str}
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

    reduce_response = call_extraction_with_fallback(
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

    raise RuntimeError("Failed to obtain valid specification output from the configured providers.")


extract_document_spec = run_extraction_pipeline