# backend/api/routes.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from backend.api.models import UploadResponse, AskRequest, AskResponse
from backend.nlp_modules import universal_parser, embedder, vector_store, summarizer
from backend.nlp_modules import risk_rules
from pathlib import Path
import pickle

router = APIRouter()

DATA_DIR   = Path("data/legal_docs/contracts")
INDEX_PATH = Path("vector_db/contracts.faiss").resolve()
META_PATH  = Path("vector_db/contracts_text.pkl").resolve()
EMBED_DIM  = 384  # MiniLM

# ──────────────────────────────────────────────────────────────
# Lazy‑load FAISS index
_db_cache = None
def get_db():
    global _db_cache
    if _db_cache is None:
        if not INDEX_PATH.exists():
            raise HTTPException(500, "Index not built yet. Upload a file first.")
        blob = pickle.load(META_PATH.open("rb"))          # {"texts": [...], "metas": [...]}
        _db_cache = vector_store.VectorDB.load(
            INDEX_PATH,
            EMBED_DIM,
            texts = blob["texts"],
            metas = blob["metas"],
        )
    return _db_cache
# ──────────────────────────────────────────────────────────────



@router.post("/upload", response_model=UploadResponse)
async def upload_contract(file: UploadFile = File(...)):
    # 1) ── save temp file ──────────────────────────────────────
    dest = DATA_DIR / file.filename
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as fh:
        fh.write(await file.read())
    await file.close()

    # 2) ── extract raw text & run risk engine ─────────────────
    text  = universal_parser.extract_text(str(dest))
    risks = risk_rules.detect_risks(text)          # optional ‑‐ keep

    # 3) ── chunk + embed ──────────────────────────────────────
    chunks, metas = universal_parser.split_into_chunks(text)
    if not chunks:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, "No text extracted from file.")
    for m in metas:
        m["source"] = file.filename
        m["risks"]  = risks

    embeddings = embedder.embed_chunks(chunks)

    # 4) ── add to / build index on disk ───────────────────────
    db = get_db() if INDEX_PATH.exists() else vector_store.VectorDB()
    db.add(embeddings, chunks, metas)
    db.save(INDEX_PATH)
    pickle.dump({"texts": db.text_chunks, "metas": db.meta_chunks},
                META_PATH.open("wb"))
    global _db_cache
    _db_cache = db                               # hot‑reload cache

    # 5) ── LLM summary of the whole doc (first ~2 chunks) ────
    ctx_for_llm = "\n".join(chunks[:2])          # ~600 tokens raw
    summary = summarizer.summarize_with_groq(
        "Provide a concise, professional summary of this document.",
        ctx_for_llm,
    )

    # 6) ── clean up temp file & respond ───────────────────────
    dest.unlink(missing_ok=True)
    return {
        "filename":     file.filename,
        "chunks_added": len(chunks),
        "risks":        risks,
        "summary":      summary,                  # ← NEW
    }

@router.post("/ask", response_model=AskResponse)
async def ask_question(payload: AskRequest):
    db = get_db()
    q_emb = embedder.embed_chunks([payload.query])[0]

    # Hybrid search: vectors + BM25 (falls back to .search if not implemented)
    if hasattr(db, "hybrid_search"):
        raw = db.hybrid_search(payload.query, q_emb, top_k=20)
    else:
        raw = db.search(q_emb, top_k=20)

    # Optional source filter
    if payload.source:
        raw = [r for r in raw if r["meta"]["source"] == payload.source]

    # De‑duplicate while preserving order
    seen, top = set(), []
    for r in raw:
        key = (r["meta"]["clause"], r["meta"]["source"], r["text"][:120])
        if key not in seen:
            seen.add(key)
            top.append(r)
        if len(top) == 6:
            break

    context = "\n".join(r["text"] for r in top)
    answer = summarizer.summarize_with_groq(f"Answer the question clearly and cite relevant legal clauses. Question: {payload.query}", context)

    citations = [
        {
            "clause":  r["meta"]["clause"][:80],
            "source":  r["meta"]["source"],
            "snippet": r["text"][:160] + "…",
        }
        for r in top
    ]

    return {"answer": answer, "citations": citations}
