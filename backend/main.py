import os
import sys
import logging
from dotenv import load_dotenv

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

for path in [CURRENT_DIR, ROOT_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from deps import LOG_FILE
from routers.phase1_router import router as phase1_router
from routers.phase2_router import router as phase2_router
from routers.logs_router import router as logs_router

logger = logging.getLogger(__name__)


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
    for noisy in ["httpx", "transformers", "urllib3", "sentence_transformers", "chromadb"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)


def clear_generated_workspace(base_dir: str = "generated_src") -> str:
    """Deletes and recreates the generated source workspace directory."""
    import shutil

    abs_dir = os.path.abspath(base_dir)
    if os.path.exists(abs_dir):
        shutil.rmtree(abs_dir, ignore_errors=True)
    os.makedirs(abs_dir, exist_ok=True)
    logger.info(f"Cleared and recreated generated workspace at {abs_dir}")
    return abs_dir


configure_logging()

app = FastAPI(title="AI-Native Core API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(phase1_router)
app.include_router(phase2_router)
app.include_router(logs_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)