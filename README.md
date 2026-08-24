# AI-Native: Requirements RAG Pipeline (Phase 1)

A modular Retrieval-Augmented Generation engine designed to parse multi-format software specification documents (PDF, TXT, MD), index chunks using BAAI/bge-small-en-v1.5, and synthesize exact code constraints via Gemini 2.5 Flash.

## Modular Architecture
- `phase1_rag/parser.py`: Multi-format text extraction & recursive chunking.
- `phase1_rag/embedder.py`: ChromaDB persistent indexing & BGE-Small retrieval.
- `phase1_rag/extractor.py`: LLM requirement extraction for Phase 2 Coder Agent.
- `run_phase1.py`: Full pipeline execution entrypoint.