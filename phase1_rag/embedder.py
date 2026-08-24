import os
import chromadb
from chromadb.utils import embedding_functions
from phase1_rag.parser import extract_text_from_file, chunk_document_text

class DocumentEmbedder:
    def __init__(self, db_dir: str = "./chroma_db_docs"):
        # BAAI/bge-small-en-v1.5 provides high precision for small technical chunks
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="BAAI/bge-small-en-v1.5"
        )
        self.client = chromadb.PersistentClient(path=db_dir)
        self.collection = self.client.get_or_create_collection(
            name="requirements_specs",
            embedding_function=self.embedding_fn
        )

    def ingest_docs_folder(self, docs_dir: str = "./docs") -> str:
        """Processes all docs in directory and stores vector representations."""
        if not os.path.exists(docs_dir):
            os.makedirs(docs_dir, exist_ok=True)
            return f"Created folder '{docs_dir}'. Place specification documents here."

        documents, metadatas, ids = [], [], []
        file_count = 0

        for file in os.listdir(docs_dir):
            file_path = os.path.join(docs_dir, file)
            if os.path.isfile(file_path) and file.endswith((".pdf", ".txt", ".md")):
                file_count += 1
                raw_text = extract_text_from_file(file_path)
                chunks = chunk_document_text(raw_text)

                for idx, chunk in enumerate(chunks):
                    documents.append(chunk)
                    metadatas.append({"source": file, "chunk_id": idx})
                    ids.append(f"{file}_chunk_{idx}")

        if documents:
            self.collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
            return f"Indexed {len(documents)} total chunk(s) across {file_count} document(s)."
        return "No valid PDF, TXT, or MD files found in docs folder."

    def retrieve_relevant_chunks(self, query: str, top_k: int = 3) -> list[dict]:
        """Queries database for top matching semantic chunks."""
        results = self.collection.query(query_texts=[query], n_results=top_k)
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        return [{"text": doc, "source": meta["source"]} for doc, meta in zip(docs, metas)]