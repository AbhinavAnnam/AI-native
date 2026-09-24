import os
import logging
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException

from models.schemas import ProjectSpecs
from phase1_rag.extractor import run_extraction_pipeline
from deps import get_embedder, get_chroma_client, clear_log_file

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/phase1", tags=["Phase 1 Extractor"])


@router.post("/extract")
async def extract_specifications(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
    
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is missing from environment or .env file.",
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
        )
        return {"status": "success", "specs": specs_dict}
    except Exception as exc:
        logger.error(f"Error during Phase 1 pipeline execution: {str(exc)}")
        raise HTTPException(status_code=500, detail=str(exc))