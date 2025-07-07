import fitz  # PyMuPDF
import cv2
import pytesseract
import numpy as np


def extract_text_from_pdf(path: str) -> str:
    """Extract text from a native (searchable) PDF."""
    text_parts = []
    # Use a context‑manager so the document auto‑closes
    with fitz.open(path) as doc:
        for page in doc:
            text_parts.append(page.get_text())
    return "\n".join(text_parts)


def extract_text_from_scanned_pdf(path: str) -> str:
    """OCR every page of a scanned PDF."""
    text_parts = []
    with fitz.open(path) as doc:            # auto‑close here too
        for page in doc:
            pix = page.get_pixmap()
            img = cv2.imdecode(
                np.frombuffer(pix.tobytes(), dtype=np.uint8), 1
            )
            if img is not None:
                text_parts.append(pytesseract.image_to_string(img))
    return "\n".join(text_parts)
