import os
from dotenv import load_dotenv
from phase1_rag.embedder import DocumentEmbedder
from phase1_rag.extractor import RequirementExtractor

load_dotenv()

def main():
    print("--- Step 1: Indexing Documents ---")
    embedder = DocumentEmbedder()
    status = embedder.ingest_docs_folder("./docs")
    print(status)

    print("\n--- Step 2: Extracting Synthesized Context ---")
    extractor = RequirementExtractor()
    feature_task = "Implement order discount logic and handle validation errors"
    
    context = extractor.synthesize_requirements(feature_task)

    print("\n=================== PAYLOAD FOR CODER AGENT ===================")
    print(context)

if __name__ == "__main__":
    main()