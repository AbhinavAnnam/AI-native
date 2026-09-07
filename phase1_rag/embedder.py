import logging
import os
import chromadb
from dotenv import load_dotenv
from google import genai

load_dotenv()


class DocumentEmbedder:

    def __init__(
        self,
        db_dir: str = "chroma_db_docs",
        collection_name: str = "active_rag_context",
    ):
        self.db_dir = db_dir
        self.collection_name = collection_name
        self.chroma_client = chromadb.PersistentClient(path=self.db_dir)

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. Ensure it is set in your .env file."
            )
        self.gemini_client = genai.Client(api_key=api_key)

    def _get_embeddings(
        self, texts: list[str], batch_size: int = 50
    ) -> list[list[float]]:
        """Batches texts to call active gemini-embedding-001 without hitting request limits."""
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self.gemini_client.models.embed_content(
                model="gemini-embedding-001",
                contents=batch,
            )
            all_embeddings.extend([e.values for e in response.embeddings])

        return all_embeddings

    def ingest_chunks(self, chunks: list[dict]):
        """Clears existing vector collection and ingests fresh embeddings into ChromaDB."""
        try:
            self.chroma_client.delete_collection(name=self.collection_name)
            logging.info(
                f"Cleared existing '{self.collection_name}' collection."
            )
        except Exception:
            logging.info(
                f"Initialized new '{self.collection_name}' collection."
            )

        collection = self.chroma_client.create_collection(
            name=self.collection_name
        )

        texts = [c["text"] for c in chunks]
        metadatas = [
            {"source": c.get("source", "doc"), "page": c.get("page", 1)}
            for c in chunks
        ]
        ids = [f"doc_chunk_{i}" for i in range(len(chunks))]

        embeddings = self._get_embeddings(texts)

        collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )
        logging.info(
            f"Successfully indexed {len(chunks)} chunks into ChromaDB."
        )

    def retrieve_relevant_chunks(
        self, query: str, top_k: int = 20
    ) -> list[dict]:
        collection = self.chroma_client.get_collection(
            name=self.collection_name
        )
        query_embedding = self._get_embeddings([query])

        results = collection.query(
            query_embeddings=query_embedding, n_results=top_k
        )

        retrieved_chunks = []
        if results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            for doc, meta in zip(docs, metas):
                retrieved_chunks.append({
                    "text": doc,
                    "source": meta.get("source", "Unknown"),
                    "page": meta.get("page", 1),
                })
        return retrieved_chunks