import os
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

DOCS_DIR = "./docs"
CHROMA_DB_DIR = "./chroma_db_docs"

print("🧠 Loading local embedding model (all-MiniLM-L6-v2)...")
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# 2. Connect to Local Persistent Vector Store
client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
collection = client.get_or_create_collection(
    name="documentation_rag",
    embedding_function=embedding_fn
)

doc_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size=300,
    chunk_overlap=40
)

def ingest_docs():
    """Reads docs/ folder, splits markdown files into chunks, and stores vectors."""
    if not os.path.exists(DOCS_DIR):
        print(f"❌ Error: Directory '{DOCS_DIR}' does not exist. Create it first.")
        return

    documents, metadatas, ids = [], [], []
    chunk_counter = 0

    for root, _, files in os.walk(DOCS_DIR):
        for file in files:
            if file.endswith((".md", ".txt")):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Split document into chunks
                chunks = doc_splitter.split_text(content)
                for idx, chunk in enumerate(chunks):
                    documents.append(chunk)
                    metadatas.append({"file_name": file, "chunk_id": idx})
                    ids.append(f"{file}_{idx}")
                    chunk_counter += 1

    if documents:
        collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"✅ Ingestion successful! Indexed {chunk_counter} chunk(s) into ChromaDB.")
    else:
        print(f"⚠️ No .md or .txt files found in '{DOCS_DIR}'.")

def query_docs(query_text: str, top_k: int = 1):
    """Executes semantic vector search against stored documentation."""
    print(f"\n🔍 Search Query: '{query_text}'")
    results = collection.query(
        query_texts=[query_text],
        n_results=top_k
    )
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    if not docs:
        print("No matching documentation found.")
        return
    for i, (doc, meta) in enumerate(zip(docs, metas)):
        print(f"--- [Match #{i+1} | Source: {meta['file_name']} | Chunk: {meta['chunk_id']}] ---")
        print(doc.strip())

if __name__ == "__main__":
    ingest_docs()
    query_docs("What exception is thrown if the shopping cart total is negative or zero?")
    query_docs("How is the price calculated for a PLATINUM tier member spending over 100 dollars?")
    query_docs("Where should transaction error logs be written when a failure occurs?")