import os
import logging
from pypdf import PdfReader

try:
    import docx
except ImportError:
    docx = None


class PDFDirectoryParser:
    """Parses PDF, Markdown (.md), Text (.txt), and Word (.docx) files from a directory."""

    def __init__(self, directory_path: str):
        self.directory_path = directory_path

    def parse_all(self) -> list[dict]:
        chunks = []
        if not os.path.exists(self.directory_path):
            logging.warning(f"Directory '{self.directory_path}' does not exist.")
            return chunks

        for filename in os.listdir(self.directory_path):
            file_path = os.path.join(self.directory_path, filename)
            ext = filename.lower().split(".")[-1]

            if ext == "pdf":
                chunks.extend(self._parse_pdf(file_path, filename))
            elif ext in ["md", "markdown", "txt"]:
                chunks.extend(self._parse_text_or_md(file_path, filename))
            elif ext == "docx":
                chunks.extend(self._parse_docx(file_path, filename))

        logging.info(f"Successfully parsed {len(chunks)} document chunks from {self.directory_path}")
        return chunks

    def _parse_pdf(self, file_path: str, filename: str) -> list[dict]:
        pdf_chunks = []
        try:
            reader = PdfReader(file_path)
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                if text and text.strip():
                    pdf_chunks.append({
                        "text": text.strip(),
                        "source": filename,
                        "page": page_num
                    })
        except Exception as e:
            logging.error(f"Failed to parse PDF '{filename}': {e}")
        return pdf_chunks

    def _parse_text_or_md(self, file_path: str, filename: str) -> list[dict]:
        text_chunks = []
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            if content.strip():
                # Split Markdown/Text by double newlines to form logical chunk sections
                sections = [s.strip() for s in content.split("\n\n") if s.strip()]
                for sec_num, section in enumerate(sections, start=1):
                    text_chunks.append({
                        "text": section,
                        "source": filename,
                        "page": sec_num
                    })
        except Exception as e:
            logging.error(f"Failed to parse Markdown/Text file '{filename}': {e}")
        return text_chunks

    def _parse_docx(self, file_path: str, filename: str) -> list[dict]:
        docx_chunks = []
        if docx is None:
            logging.warning(f"python-docx not installed. Install with 'pip install python-docx' to parse {filename}")
            return docx_chunks
        try:
            doc = docx.Document(file_path)
            paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            full_text = "\n\n".join(paragraphs)
            if full_text.strip():
                sections = [s.strip() for s in full_text.split("\n\n") if s.strip()]
                for sec_num, section in enumerate(sections, start=1):
                    docx_chunks.append({
                        "text": section,
                        "source": filename,
                        "page": sec_num
                    })
        except Exception as e:
            logging.error(f"Failed to parse DOCX file '{filename}': {e}")
        return docx_chunks


# Alias for multi-format backwards compatibility
DocumentDirectoryParser = PDFDirectoryParser