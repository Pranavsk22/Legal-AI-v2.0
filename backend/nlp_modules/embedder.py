from sentence_transformers import SentenceTransformer
import os

os.environ["HF_HOME"] = "/home/user/.cache"
os.environ["TRANSFORMERS_CACHE"] = "/home/user/.cache/transformers"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "/home/user/.cache/sentence_transformers"

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_chunks(text_chunks):
    return model.encode(text_chunks)
