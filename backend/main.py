import os
import sys
import json
import logging
from typing import List
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pypdf import PdfReader
import chromadb
from google import genai
from google.genai import types

# Load variables from .env file
load_dotenv()

LOG_FILE = "pipeline.log"

def configure_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
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

def extract_chunks_from_pdfs(files: List[UploadFile]) -> List[dict]:
    chunks = []
    for file in files:
        logger.info(f"Parsing uploaded PDF: {file.filename}")
        reader = PdfReader(file.file)
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                page_text = text.strip()
                chunk_size = 800
                overlap = 100
                for i in range(0, len(page_text), chunk_size - overlap):
                    chunk_content = page_text[i:i + chunk_size]
                    chunks.append({
                        "text": chunk_content,
                        "id": f"{file.filename}_p{page_num+1}_c{i}"
                    })
    return chunks

@app.post("/api/phase1/extract")
async def extract_specifications(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    # Validate API key presence
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500, 
            detail="GEMINI_API_KEY environment variable is missing. Please set it in your .env file."
        )

    clear_log_file()
    logger.info("API Request received: Starting Phase 1 extraction pipeline...")
    logger.info(f"Received {len(files)} uploaded document(s) for processing.")

    try:
        chunks = extract_chunks_from_pdfs(files)
        if not chunks:
            raise ValueError("No text content could be extracted from the uploaded PDF files.")
        
        logger.info(f"Successfully processed PDF into {len(chunks)} context fragments.")

        logger.info("Generating embeddings and indexing chunks in ChromaDB...")
        embedder = get_embedder()
        chroma_client = get_chroma_client()
        
        collection_name = f"temp_col_{int(os.urandom(4).hex(), 16)}"
        collection = chroma_client.create_collection(name=collection_name)

        texts = [c["text"] for c in chunks]
        ids = [c["id"] for c in chunks]
        embeddings = embedder.encode(texts).tolist()

        collection.add(
            documents=texts,
            embeddings=embeddings,
            ids=ids
        )

        logger.info("Retrieving top relevant chunks from vector store...")
        query = "Extract feature specifications, python file module names, business rules, function signatures, and error exceptions."
        query_embedding = embedder.encode([query]).tolist()

        top_k = min(20, len(chunks))
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )

        retrieved_docs = results["documents"][0]
        logger.info(f"Successfully retrieved {len(retrieved_docs)} chunks. Formatting context for Gemini...")
        context_str = "\n\n---\n\n".join(retrieved_docs)

        chroma_client.delete_collection(name=collection_name)

        logger.info("Sending context to Gemini 3.6 Flash for structured extraction...")
        
        # Initialize client with explicit API key
        client = genai.Client(api_key=api_key)

        prompt = f"""
        Extract all operational feature domain specifications from the following technical PRD context.
        Ensure every domain includes business rules, target python module name, detailed function signatures with type hints, and exception lists.

        Technical Context:
        {context_str}
        """

        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ProjectSpecs,
                temperature=0.1
            )
        )

        logger.info("Gemini specification synthesis completed successfully.")
        logger.info("Phase 1 extraction finished successfully.")

        parsed_specs = json.loads(response.text)
        return {"status": "success", "specs": parsed_specs}

    except Exception as e:
        logger.error(f"Error during Phase 1 pipeline execution: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/logs")
async def get_logs():
    if not os.path.exists(LOG_FILE):
        return {"logs": "No log file found."}
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            logs = f.read()
        return {"logs": logs if logs.strip() else "Log file is empty."}
    except Exception as e:
        return {"logs": f"Error reading log file: {str(e)}"}