import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_text_from_file(file_path: str) -> str:
    """Reads PDF, TXT, or MD files and returns raw text."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        reader = PdfReader(file_path)
        pages = [
            f"--- Page {i+1} ---\n{page.extract_text()}"
            for i, page in enumerate(reader.pages)
            if page.extract_text()
        ]
        return "\n".join(pages)
    elif ext in [".txt", ".md"]:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported extension: {ext}")

def chunk_document_text(text: str, chunk_size: int = 500, chunk_overlap: int = 80) -> list[str]:
    """Splits text into small overlapping chunks to keep context sharp."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n# ", "\n\n", "\n", " ", ""]
    )
    return splitter.split_text(text)