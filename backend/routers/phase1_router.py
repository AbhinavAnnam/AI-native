import os
import logging
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException

from models.schemas import ProjectSpecs
from phase1_rag.extractor import run_extraction_pipeline
from phase1_rag.providers import fallback_keys_available
from deps import get_embedder, get_chroma_client, clear_log_file

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/phase1", tags=["Phase 1 Extractor"])


@router.post("/extract")
async def extract_specifications(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
    
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("gemini_api_key") or ""
    if not gemini_key and not fallback_keys_available():
        raise HTTPException(
            status_code=500,
            detail="Set GROQ_API_KEY, OPENROUTER_API_KEY, or GEMINI_API_KEY in .env.",
        )

    clear_log_file()
    logger.info("API Request received: Starting Phase 1 extraction pipeline...")
    logger.info(f"Received {len(files)} uploaded document(s) for processing.")

    try:
        specs_dict = run_extraction_pipeline(
            files=files,
            gemini_key=gemini_key,
            embedder=get_embedder(),
            chroma_client=get_chroma_client(),
            response_schema=ProjectSpecs,
            max_parallel_workers=1,
        )
        return {"status": "success", "specs": specs_dict}
    except Exception as exc:
        logger.error(f"Error during Phase 1 pipeline execution: {str(exc)}")
        raise HTTPException(status_code=500, detail=str(exc))
