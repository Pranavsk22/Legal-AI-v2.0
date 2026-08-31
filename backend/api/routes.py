# backend/api/routes.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from backend.api.models import UploadResponse, AskRequest, AskResponse, DocumentDraftResponse, DocumentConfirmRequest
from backend.nlp_modules import universal_parser, embedder, vector_store, summarizer
from backend.nlp_modules import risk_rules
from backend.nlp_modules.schema import DocumentRecord
from pathlib import Path
import pickle
from typing import List, Optional

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

    # 3) ── chunk + metadata extraction + schema validation ─────
    doc_meta = universal_parser.extract_metadata(text)
    source_format = Path(file.filename).suffix.upper().replace(".", "") or "TXT"

    chunks, basic_metas = universal_parser.split_into_chunks(text)
    if not chunks:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, "No text extracted from file.")

    validated_metas = []
    for i, chunk in enumerate(chunks):
        chunk_risks = risk_rules.detect_risks(chunk)
        try:
            record = DocumentRecord(
                doc_id=file.filename,
                doc_type=doc_meta["doc_type"],
                risk_flags=chunk_risks,
                parties=doc_meta["parties"],
                effective_date=doc_meta["effective_date"],
                governing_law=doc_meta["governing_law"],
                source_format=source_format,
                clause_index=i
            )
            # Store in dict format for vector database (making it compatible with FAISS/BM25 storage)
            # We also ensure it retains 'source' and 'risks' keys for backward compatibility in Q&A
            meta_dict = record.model_dump()
            meta_dict["source"] = file.filename
            meta_dict["risks"] = risks
            meta_dict["clause"] = basic_metas[i]["clause"]
            validated_metas.append(meta_dict)
        except Exception as e:
            dest.unlink(missing_ok=True)
            raise HTTPException(422, f"Metadata validation failed for chunk {i}: {str(e)}")

    embeddings = embedder.embed_chunks(chunks)

    # 4) ── add to / build index on disk ───────────────────────
    db = get_db() if INDEX_PATH.exists() else vector_store.VectorDB()
    
    # Run poisoning check
    poisoning_warnings = db.detect_poisoning(embeddings)
    import re
    flagged_indices = []
    for warn in poisoning_warnings:
        m = re.search(r"Chunk (\d+)", warn)
        if m:
            flagged_indices.append(int(m.group(1)))
            
    for i in range(len(validated_metas)):
        validated_metas[i]["potential_poisoning"] = (i in flagged_indices)

    db.add(embeddings, chunks, validated_metas)
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
        "poisoning_warnings": poisoning_warnings,
    }

@router.get("/search")
async def search_contracts(
    query: Optional[str] = None,
    risk_type: Optional[str] = None,
    doc_type: Optional[str] = None,
    governing_law: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
):
    try:
        db = get_db()
    except HTTPException:
        return []

    q_emb = None
    if query:
        q_emb = embedder.embed_chunks([query])[0]

    raw = db.hybrid_search(
        query=query,
        query_emb=q_emb,
        top_k=20,
        risk_type=risk_type,
        doc_type=doc_type,
        governing_law=governing_law,
        date_from=date_from,
        date_to=date_to,
    )
    return raw

@router.post("/ingest/draft", response_model=DocumentDraftResponse)
async def ingest_draft(file: UploadFile = File(...)):
    # 1) save temp file
    dest = DATA_DIR / file.filename
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as fh:
        fh.write(await file.read())
    await file.close()

    try:
        # 2) extract text
        text = universal_parser.extract_text(str(dest))
        # 3) run risk rules
        risks = risk_rules.detect_risks(text)
        # 4) extract metadata
        doc_meta = universal_parser.extract_metadata(text)
        # 5) chunk text
        chunks, _ = universal_parser.split_into_chunks(text)
        source_format = Path(file.filename).suffix.upper().replace(".", "") or "TXT"
    finally:
        # Clean up temp file
        dest.unlink(missing_ok=True)

    # Run poisoning check
    embeddings = embedder.embed_chunks(chunks)
    db = get_db() if INDEX_PATH.exists() else vector_store.VectorDB()
    poisoning_warnings = db.detect_poisoning(embeddings)

    return {
        "doc_id": file.filename,
        "doc_type": doc_meta["doc_type"],
        "risk_flags": risks,
        "parties": doc_meta["parties"],
        "effective_date": doc_meta["effective_date"],
        "governing_law": doc_meta["governing_law"],
        "source_format": source_format,
        "chunks": chunks,
        "poisoning_warnings": poisoning_warnings
    }

@router.post("/ingest/confirm")
async def ingest_confirm(payload: DocumentConfirmRequest):
    # 1) Validate metadata using Pydantic schema for each chunk
    validated_metas = []
    for i, chunk in enumerate(payload.chunks):
        chunk_risks = risk_rules.detect_risks(chunk)
        try:
            record = DocumentRecord(
                doc_id=payload.doc_id,
                doc_type=payload.doc_type,
                risk_flags=chunk_risks,
                parties=payload.parties,
                effective_date=payload.effective_date,
                governing_law=payload.governing_law,
                source_format=payload.source_format,
                clause_index=i
            )
            meta_dict = record.model_dump()
            meta_dict["source"] = payload.doc_id
            meta_dict["risks"] = payload.risk_flags
            meta_dict["clause"] = f"Clause {i}"
            validated_metas.append(meta_dict)
        except Exception as e:
            raise HTTPException(422, f"Metadata validation failed for chunk {i}: {str(e)}")

    # 2) Embed chunks
    embeddings = embedder.embed_chunks(payload.chunks)

    # 3) Write to vector store
    db = get_db() if INDEX_PATH.exists() else vector_store.VectorDB()
    
    # Run poisoning check
    poisoning_warnings = db.detect_poisoning(embeddings)
    import re
    flagged_indices = []
    for warn in poisoning_warnings:
        m = re.search(r"Chunk (\d+)", warn)
        if m:
            flagged_indices.append(int(m.group(1)))
            
    for i in range(len(validated_metas)):
        validated_metas[i]["potential_poisoning"] = (i in flagged_indices)

    db.add(embeddings, payload.chunks, validated_metas)
    db.save(INDEX_PATH)
    
    pickle.dump({"texts": db.text_chunks, "metas": db.meta_chunks},
                 META_PATH.open("wb"))
    global _db_cache
    _db_cache = db                               # hot-reload cache

    return {
        "status": "success",
        "doc_id": payload.doc_id,
        "chunks_indexed": len(payload.chunks)
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

    # Verify citations and claims
    validation_status, validation_details = summarizer.verify_citations(answer, top)

    citations = [
        {
            "clause":  r["meta"]["clause"][:80],
            "source":  r["meta"]["source"],
            "snippet": r["text"][:160] + "…",
        }
        for r in top
    ]

    return {
        "answer": answer,
        "citations": citations,
        "validation_status": validation_status,
        "citation_validation_details": validation_details
    }
