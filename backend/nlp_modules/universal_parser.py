import re
import pathlib
from .pdf_parser import extract_text_from_pdf, extract_text_from_scanned_pdf
from .docx_parser import extract_text_from_docx
from .adoc_parser import extract_text_from_adoc
from .html_parser import extract_text_from_html

HEADING_RE = re.compile(r"^(?:section|clause|article)\s+([\dA-Za-z.\-()]+).*", re.I)

def extract_text(path: str) -> str:
    ext = pathlib.Path(path).suffix.lower()
    
    if ext == ".pdf":
        text = extract_text_from_pdf(path)
        if not text.strip():
            print("⚠️ PDF has no text. Trying OCR...")
            return extract_text_from_scanned_pdf(path)
        return text
    
    elif ext == ".docx":
        return extract_text_from_docx(path)
    
    elif ext == ".adoc":
        return extract_text_from_adoc(path)
        
    elif ext == ".txt":
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
            
    elif ext in (".html", ".htm"):
        return extract_text_from_html(path)
    
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def split_into_chunks(text: str, chunk_size=300, overlap=50):
    words, chunks, metas = text.split(), [], []
    clause = "Unknown"
    i = 0
    while i < len(words):
        window = words[i : i + chunk_size]
        chunk_text = " ".join(window)

        # detect heading inside the window’s first 20 words
        head_search = HEADING_RE.search(" ".join(window[:20]))
        if head_search:
            clause = head_search.group(0).strip()

        chunks.append(chunk_text)
        metas.append({"clause": clause})        # 👈 store current clause
        i += chunk_size - overlap
    return chunks, metas