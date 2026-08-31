import json
from phase1_rag.parser import parse_documents
from phase1_rag.embedder import DocumentEmbedder
from phase1_rag.extractor import RequirementExtractor

def main():
    print("--- Step 1: Parsing Documents ---")
    chunks = parse_documents("./docs")
    if not chunks:
        print("No documents found in ./docs folder.")
        return

    print("--- Step 2: Indexing Chunks into Vector DB ---")
    embedder = DocumentEmbedder()
    status = embedder.ingest_chunks(chunks)
    print(status)

    print("--- Step 3: Synthesizing Feature Requirements ---")
    extractor = RequirementExtractor()
    feature_task = "Extract all system features, requirements, function signatures, and exceptions from the documentation."
    
    # Get raw JSON string from Gemini
    raw_json_str = extractor.synthesize_requirements(feature_task)
    
    # Parse and format JSON with indentation
    parsed_json = json.loads(raw_json_str)
    pretty_json = json.dumps(parsed_json, indent=4)
    
    # Write formatted output to file
    with open("extracted_specs.json", "w", encoding="utf-8") as f:
        f.write(pretty_json)
        
    print("\n=================== EXTRACTED JSON SPECS ===================")
    print(pretty_json)
    print("\nSuccessfully saved output to extracted_specs.json!")

if __name__ == "__main__":
    main()