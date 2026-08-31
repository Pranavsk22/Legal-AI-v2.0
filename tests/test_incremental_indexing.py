# tests/test_incremental_indexing.py
import pytest
import numpy as np
from backend.nlp_modules.vector_store import VectorDB
from backend.nlp_modules import embedder
from unittest.mock import patch

def test_incremental_indexing():
    # 1. Create a VectorDB instance
    db = VectorDB(dim=384)
    
    # 2. Prepare 10 documents
    texts = [f"This is document number {i} containing specific clause details." for i in range(10)]
    embeddings = embedder.embed_chunks(texts)
    
    metas = [{
        "doc_id": f"doc_{i}",
        "doc_type": "Agreement",
        "risk_flags": [],
        "parties": "Unknown",
        "effective_date": "2026-01-01",
        "governing_law": "Delaware",
        "source_format": "TXT",
        "clause_index": 0,
        "source": f"doc_{i}",
        "clause": "Unknown"
    } for i in range(10)]
    
    # Add first 10 docs
    db.add(embeddings, texts, metas)
    assert len(db.text_chunks) == 10
    assert db.index.ntotal == 10
    
    # 3. Prepare 11th document (incrementally added)
    text_11 = "This is the eleventh unique document containing a secret keyword: pineapple."
    emb_11 = embedder.embed_chunks([text_11])
    
    # Patch embedder to make sure it is not called during add_document
    with patch("backend.nlp_modules.embedder.embed_chunks") as mock_embed:
        db.add_document("doc_11", [text_11], emb_11)
        mock_embed.assert_not_called()
        
    # Check that document count is 11
    assert len(db.text_chunks) == 11
    assert db.index.ntotal == 11
    
    # 4. Search and verify the 11th document is retrieved
    q_emb = embedder.embed_chunks(["pineapple"])[0]
    results = db.search(q_emb, top_k=1)
    
    assert len(results) == 1
    assert "pineapple" in results[0]["text"]
    assert results[0]["meta"]["doc_id"] == "doc_11"
