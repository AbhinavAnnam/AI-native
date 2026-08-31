import os
from pypdf import PdfReader

def parse_documents(docs_dir: str = "./docs") -> list[dict]:
    """Parses all documents in a folder into page-level chunks with metadata."""
    chunks = []
    if not os.path.exists(docs_dir):
        return chunks

    for file in os.listdir(docs_dir):
        file_path = os.path.join(docs_dir, file)
        if not os.path.isfile(file_path):
            continue

        # Page-by-page PDF extraction
        if file.endswith(".pdf"):
            reader = PdfReader(file_path)
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                if text and text.strip():
                    chunks.append({
                        "text": text.strip(),
                        "metadata": {"source": file, "page": page_num}
                    })

        # Section-by-section TXT / MD extraction
        elif file.endswith((".txt", ".md")):
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            sections = [s.strip() for s in text.split("\n\n") if s.strip()]
            for idx, sec in enumerate(sections, start=1):
                chunks.append({
                    "text": sec,
                    "metadata": {"source": file, "section": idx}
                })

    return chunks