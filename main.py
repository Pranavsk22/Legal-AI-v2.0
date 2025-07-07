from backend.nlp_modules.pdf_parser import extract_text_from_pdf
from backend.nlp_modules.embedder import embed_chunks
from backend.nlp_modules.vector_store import VectorDB
from backend.nlp_modules.summarizer import summarize_with_groq
from scripts.index_contracts import build_index
from scripts.summarize_query import ask

#from sentence_transformers.util import batch_to_device

def split_into_chunks(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

if __name__ == "__main__":
    path = "data/legal_docs/contracts/G_Narayanaswami_Naidu_vs_C_Krishnamurthi_And_Anr_on_23_January_1958.PDF"  # Replace with actual
    text = extract_text_from_pdf(path)
    chunks = split_into_chunks(text)

    embeddings = embed_chunks(chunks)

    db = VectorDB()
    db.add(embeddings, chunks)

    query = "Termination conditions of the contract?"
    query_emb = embed_chunks([query])[0]

    relevant_chunks = db.search(query_emb)
    context = "\n".join(relevant_chunks)

    summary = summarize_with_groq(context)
    print("\n📘 Summary of Retrieved Context:\n", summary)
    
    build_index()               # 1) build / refresh index
    ans, _ = ask("Summarize the termination clauses.")  # 2) demo query
    print("\n🔔 Quick demo answer:\n", ans)