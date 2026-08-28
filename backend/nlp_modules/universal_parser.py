import re
import pathlib
from .pdf_parser import extract_text_from_pdf, extract_text_from_scanned_pdf
from .docx_parser import extract_text_from_docx
from .adoc_parser import extract_text_from_adoc
from .html_parser import extract_text_from_html
from .schema import DocumentRecord
from . import summarizer

HEADING_RE = re.compile(r"^(?:section|clause|article)\s+([\dA-Za-z.\-()]+).*", re.I)

def extract_text(path: str) -> str:
    ext = pathlib.Path(path).suffix.lower()
    
    if ext == ".pdf":
        text = extract_text_from_pdf(path)
        if not text.strip():
            print("Warning: PDF has no text. Trying OCR...")
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


def extract_metadata_fallback(text: str) -> dict:
    """Fallback regex-based metadata extraction when Groq fails."""
    # 1. doc_type
    doc_type = "Unknown"
    first_lines = "\n".join(text.splitlines()[:15]).lower()
    if "lease" in first_lines or "rental" in first_lines:
        doc_type = "Lease Agreement"
    elif "non-disclosure" in first_lines or "nda" in first_lines or "confidentiality" in first_lines:
        doc_type = "NDA"
    elif "service agreement" in first_lines or "services agreement" in first_lines:
        doc_type = "Service Agreement"
    elif "employment" in first_lines:
        doc_type = "Employment Agreement"
    elif "agreement" in first_lines:
        doc_type = "Agreement"
    elif "contract" in first_lines:
        doc_type = "Contract"

    # 2. parties
    parties = []
    match_between = re.search(r"between\s+([\w\s,]+?)\s+and\s+([\w\s,]+?)(?:,|\.|agree|entered)", text[:1500], re.I)
    if match_between:
        p1 = match_between.group(1).strip()
        p2 = match_between.group(2).strip()
        p1 = re.sub(r"\s+", " ", p1)
        p2 = re.sub(r"\s+", " ", p2)
        parties = [p1, p2]
    else:
        parties = ["Unknown"]

    # 3. effective_date
    effective_date = None
    match_date = re.search(r"(?:effective\s+(?:as\s+of\s+)?|date\s+is\s+|on\s+)(?:the\s+)?(\d{1,2}(?:st|nd|rd|th)?\s+(?:day\s+of\s+)?(?:[A-Z][a-z]+),\s+\d{4}|\w+\s+\d{1,2},\s+\d{4}|\d{4}-\d{2}-\d{2})", text[:1500], re.I)
    if match_date:
        effective_date = match_date.group(1).strip()
    else:
        match_date2 = re.search(r"\b([A-Z][a-z]+)\s+(\d{1,2}),\s+(\d{4})\b", text[:1500])
        if match_date2:
            effective_date = f"{match_date2.group(1)} {match_date2.group(2)}, {match_date2.group(3)}"

    # 4. governing_law
    governing_law = "Unknown"
    match_law = re.search(r"governed\s+by\s+and\s+construed\s+in\s+accordance\s+with\s+(?:the\s+)?laws\s+of\s+(?:the\s+)?(?:state\s+of\s+)?([A-Z][a-zA-Z\s,]+?)(?:\.|\bclause\b|\band\b|;)", text, re.I)
    if not match_law:
        match_law = re.search(r"laws?\s+of\s+(?:the\s+)?(?:state\s+of\s+)?([A-Z][a-zA-Z\s]+?)(?:\s+governs|\s+shall\s+govern|\band\b|\.|\bcourt\b)", text, re.I)
    if match_law:
        governing_law = match_law.group(1).strip()
        governing_law = re.sub(r"\s+", " ", governing_law)

    return {
        "doc_type": doc_type,
        "parties": parties,
        "effective_date": effective_date,
        "governing_law": governing_law
    }


def extract_metadata(text: str) -> dict:
    """Extract metadata trying Groq first, then falling back to regex."""
    meta = summarizer.extract_metadata_with_groq(text)
    
    # Check if Groq returned valid metadata
    is_valid = any(meta.get(k) is not None for k in ["doc_type", "parties", "effective_date", "governing_law"])
    if not is_valid:
        meta = extract_metadata_fallback(text)
        
    return {
        "doc_type": meta.get("doc_type") or "Unknown",
        "parties": meta.get("parties") or ["Unknown"],
        "effective_date": meta.get("effective_date"),
        "governing_law": meta.get("governing_law") or "Unknown"
    }


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