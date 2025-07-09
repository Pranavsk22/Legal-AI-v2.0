# backend/nlp_modules/embedder.py
from sentence_transformers import SentenceTransformer
import os, pathlib

# ---- guarantee writable cache dir ----
CACHE_BASE = "/tmp/huggingface_cache"        # <– always writable
for sub in ("", "transformers", "sentence-transformers"):
    pathlib.Path(f"{CACHE_BASE}/{sub}").mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"]                   = CACHE_BASE
os.environ["TRANSFORMERS_CACHE"]        = f"{CACHE_BASE}/transformers"
os.environ["SENTENCE_TRANSFORMERS_HOME"]= f"{CACHE_BASE}/sentence-transformers"

# ---- load model (will now cache into /tmp/…) ----
model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_chunks(text_chunks):
    return model.encode(text_chunks)
