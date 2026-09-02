import os
import shutil
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
import logging

from phase1_rag.parser import PDFDirectoryParser
from phase1_rag.embedder import DocumentEmbedder
from phase1_rag.extractor import RequirementExtractor

app = FastAPI(title="Coder Agent Engine API")

TEMP_UPLOAD_DIR = "temp_uploads"
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

@app.get("/")
def read_root():
    return {"status": "online", "message": "AI-Native Engine API is running."}

@app.post("/api/phase1/extract")
async def extract_requirements(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No PDF files uploaded.")
        
    logging.info(f"API Request received: Processing {len(files)} uploaded document(s)...")

    # 1. Clean temp folder and save uploaded files
    for filename in os.listdir(TEMP_UPLOAD_DIR):
        file_path = os.path.join(TEMP_UPLOAD_DIR, filename)
        if os.path.isfile(file_path):
            os.unlink(file_path)

    for file in files:
        file_path = os.path.join(TEMP_UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

    # 2. Parse uploaded PDFs
    parser = PDFDirectoryParser(TEMP_UPLOAD_DIR)
    chunks = parser.parse_all()
    
    if not chunks:
        raise HTTPException(status_code=400, detail="Could not extract text from uploaded PDF files.")

    # 3. Embed chunks into generic ChromaDB collection
    embedder = DocumentEmbedder()
    embedder.ingest_chunks(chunks)

    # 4. Synthesize JSON specifications
    extractor = RequirementExtractor()
    raw_json_specs = extractor.synthesize_requirements(
        task_description="Extract complete functional modules, business rules, function signatures, and custom exceptions from the uploaded documentation."
    )

    # Save to disk for Coder Agent (Phase 2)
    with open("extracted_specs.json", "w") as f:
        f.write(raw_json_specs)

    logging.info("Phase 1 extraction from user uploads finished successfully.")
    return {"status": "success", "specs": raw_json_specs}