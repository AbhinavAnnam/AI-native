import os
import chromadb
from chromadb.utils import embedding_functions

class DocumentEmbedder:
    def __init__(self, db_dir: str = "./chroma_db_docs"):
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="BAAI/bge-small-en-v1.5"
        )
        self.client = chromadb.PersistentClient(path=db_dir)
        self.collection = self.client.get_or_create_collection(
            name="requirements_specs",
            embedding_function=self.embedding_fn
        )

    def ingest_chunks(self, chunks: list[dict]) -> str:
        """Stores text chunks alongside page/file metadata."""
        if not chunks:
            return "No valid documents found in docs folder."

        documents = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        ids = [f"{c['metadata']['source']}_c{idx}" for idx, c in enumerate(chunks)]

        self.collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
        return f"Indexed {len(chunks)} page/section chunk(s) into ChromaDB."

    def retrieve_relevant_chunks(self, query: str, top_k: int = 5) -> list[dict]:
        """Queries database for top matching semantic chunks."""
        results = self.collection.query(query_texts=[query], n_results=top_k)
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        return [{"text": doc, "source": meta.get("source"), "page": meta.get("page", 1)} 
                for doc, meta in zip(docs, metas)]