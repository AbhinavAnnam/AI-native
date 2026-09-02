import logging
import chromadb
from sentence_transformers import SentenceTransformer

class DocumentEmbedder:
    def __init__(self, db_dir: str = "chroma_db_docs", collection_name: str = "active_rag_context"):
        self.db_dir = db_dir
        self.collection_name = collection_name
        self.chroma_client = chromadb.PersistentClient(path=self.db_dir)
        self.embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        
    def ingest_chunks(self, chunks: list[dict]):
        """Wipes the active user collection and ingests fresh chunks."""
        try:
            self.chroma_client.delete_collection(name=self.collection_name)
            logging.info(f"Cleared existing '{self.collection_name}' collection for fresh document upload.")
        except Exception:
            logging.info(f"Initialized new '{self.collection_name}' collection.")

        collection = self.chroma_client.create_collection(name=self.collection_name)

        texts = [c["text"] for c in chunks]
        metadatas = [{"source": c.get("source", "doc"), "page": c.get("page", 1)} for c in chunks]
        ids = [f"doc_chunk_{i}" for i in range(len(chunks))]

        embeddings = self.embedding_model.encode(texts, show_progress_bar=False).tolist()

        collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        logging.info(f"Successfully indexed {len(chunks)} chunks into '{self.collection_name}'.")

    def retrieve_relevant_chunks(self, query: str, top_k: int = 20) -> list[dict]:
        collection = self.chroma_client.get_collection(name=self.collection_name)
        query_embedding = self.embedding_model.encode([query]).tolist()

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )

        retrieved_chunks = []
        if results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            for doc, meta in zip(docs, metas):
                retrieved_chunks.append({
                    "text": doc,
                    "source": meta.get("source", "Unknown"),
                    "page": meta.get("page", 1)
                })
        return retrieved_chunks