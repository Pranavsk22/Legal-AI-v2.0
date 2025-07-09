# backend/nlp_modules/embedder.py
import os
from pathlib import Path

# ── point all caches to a writable location inside the container ───────────────
CACHE_BASE = "/tmp/huggingface"                         # matches Dockerfile
Path(f"{CACHE_BASE}/transformers").mkdir(parents=True, exist_ok=True)
Path(f"{CACHE_BASE}/sentence-transformers").mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"]                    = CACHE_BASE
os.environ["TRANSFORMERS_CACHE"]         = f"{CACHE_BASE}/transformers"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = f"{CACHE_BASE}/sentence-transformers"

# ── import *after* env‑vars so the lib picks them up ───────────────────────────
from sentence_transformers import SentenceTransformer

# download → /tmp/huggingface/sentence-transformers/…
model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_chunks(text_chunks: list[str]):
    """Return embeddings for a list of text chunks."""
    return model.encode(text_chunks)
