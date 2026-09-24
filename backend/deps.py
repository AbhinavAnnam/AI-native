import os
import logging
import chromadb

LOG_FILE = "pipeline.log"
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


def clear_log_file():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            f.write("")
    