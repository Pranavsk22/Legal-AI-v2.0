# scripts/summarize_query.py
"""
Load the saved FAISS index + chunk metadata, accept a query from stdin,
retrieve top‑K chunks, and produce a Groq‑powered summary.
"""
import pickle
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.nlp_modules.embedder import embed_chunks
from backend.nlp_modules.vector_store import VectorDB
from backend.nlp_modules.summarizer import summarize_with_groq

INDEX_PATH = Path("vector_db/contracts.faiss")
META_PATH  = Path("vector_db/contracts_text.pkl")

def load_db():
    blob = pickle.loads(META_PATH.read_bytes())
    return VectorDB.load(INDEX_PATH, dim=384, texts=blob["texts"], metas=blob["metas"])

def ask(query, top_k=5):
    vdb = load_db()
    q_emb = embed_chunks([query])[0]
    results = vdb.search(q_emb, top_k=top_k)
    chunks = [r["text"] for r in results]
    context = "\n".join(chunks)
    answer  = summarize_with_groq(context)
    return answer, chunks

if __name__ == "__main__":
    db_ok = INDEX_PATH.exists() and META_PATH.exists()
    if not db_ok:
        raise RuntimeError("Index not found. Run scripts/index_contracts.py first.")

    query = input("🔍 Ask your legal question > ")
    answer, refs = ask(query)
    print("\n🧠 Groq‑LLM Summary:\n", answer)
    print("\n📚 Top context chunks:\n", "-"*40)
    for ch in refs:
        print("•", ch[:160].replace("\n", " "), "…\n")
