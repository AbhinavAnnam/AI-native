import os
import sys
import shutil

# Ensure root directory is in sys.path for cross-module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import logging
from typing import List
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb

from phase1_rag.extractor import run_extraction_pipeline
from phase2_coder.coder import CoderAgent
from phase2_coder.hitl_manager import hitl_queue
from mcp_tools.file_tools import MCPToolServer

LOG_FILE = "pipeline.log"


def configure_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    for noisy_lib in ["httpx", "transformers", "urllib3", "sentence_transformers", "chromadb"]:
        logging.getLogger(noisy_lib).setLevel(logging.WARNING)


configure_logging()
logger = logging.getLogger(__name__)
_embedder = None
_chroma_client = None


def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading SentenceTransformer model into memory...")
        _embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _embedder


def get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.Client()
    return _chroma_client


def clear_generated_workspace(target_dir: str = "generated_src"):
    """Deletes all stale files in generated_src prior to fresh code synthesis."""
    abs_path = os.path.abspath(target_dir)
    if os.path.exists(abs_path):
        for item in os.listdir(abs_path):
            item_path = os.path.join(abs_path, item)
            try:
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    if item.endswith(".py"):
                        os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            except Exception as exc:
                logger.error(f"Failed to delete {item_path}: {exc}")
    else:
        os.makedirs(abs_path, exist_ok=True)


class FunctionSpec(BaseModel):
    signature: str
    purpose: str


class FeatureSpec(BaseModel):
    feature_name: str
    target_module: str
    business_rules: List[str]
    functions: List[FunctionSpec]
    exceptions: List[str]


class ProjectSpecs(BaseModel):
    project_title: str
    features: List[FeatureSpec]


class HITLApprovalPayload(BaseModel):
    draft_id: str
    target_module: str
    approved_code: str
    action: str


app = FastAPI(title="AI-Native Core API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def clear_log_file():
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("")


@app.post("/api/phase1/extract")
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


@app.post("/api/phase2/generate-drafts")
async def generate_drafts(specs: ProjectSpecs):
    groq_key = os.getenv("GROQ_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")

    if not gemini_key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is required in .env file.",
        )

    if not groq_key and not openrouter_key:
        raise HTTPException(
            status_code=500,
            detail="At least one primary key (GROQ_API_KEY or OPENROUTER_API_KEY) must be provided in .env.",
        )

    logger.info("Starting Phase 2 Coder Agent draft generation...")
    
    # Wipe stale files from previous generation attempts
    clear_generated_workspace("generated_src")

    drafts = CoderAgent(
        groq_api_key=groq_key or "",
        gemini_api_key=gemini_key,
        openrouter_api_key=openrouter_key,
    ).generate_drafts_from_specs(specs.model_dump())

    return {"status": "success", "pending_reviews": drafts}


@app.post("/api/phase2/approve-and-commit")
async def approve_and_commit(payload: HITLApprovalPayload):
    if payload.action == "reject":
        hitl_queue.update_status(payload.draft_id, "REJECTED")
        return {"status": "rejected", "module": payload.target_module}
    try:
        written_file_path = MCPToolServer.write_module_to_disk(
            payload.target_module, payload.approved_code
        )
        hitl_queue.update_status(
            payload.draft_id, "COMMITTED", payload.approved_code
        )
        return {
            "status": "committed",
            "file_path": written_file_path,
            "module": payload.target_module,
        }
    except Exception as exc:
        logger.error(f"MCP Write Error: {str(exc)}")
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/logs")
async def get_logs():
    if not os.path.exists(LOG_FILE):
        return {"logs": "No log file found."}
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            logs = f.read()
        return {"logs": logs if logs.strip() else "Log file is empty."}
    except Exception as exc:
        return {"logs": f"Error reading log file: {str(exc)}"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)