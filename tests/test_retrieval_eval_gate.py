# tests/test_retrieval_eval_gate.py
import pytest
from io import BytesIO
from fastapi.testclient import TestClient
from backend.api.main import app
from backend.nlp_modules.vector_store import VectorDB
from backend.nlp_modules import embedder, summarizer
from scripts.eval_retrieval import CLAUSES, TEST_QUERIES, evaluate_query_list

def test_retrieval_performance_regression_gate():
    """
    CI Regression Gate: Ensures retrieval accuracy does not regress below minimum thresholds.
    Thresholds:
    - Recall@3 >= 0.85
    - MRR >= 0.80
    """
    db = VectorDB(dim=384)
    # Take a 30-clause subset for fast CI execution
    eval_clauses = CLAUSES[:30]
    embeddings = embedder.embed_chunks(eval_clauses)
    metas = [{
        "doc_id": "test_contract.txt",
        "doc_type": "Agreement",
        "risk_flags": [],
        "parties": "Parties",
        "effective_date": "2024-01-01",
        "governing_law": "Standard",
        "source_format": "TXT",
        "clause_index": i,
        "source": "test_contract.txt",
        "clause": clause.split(":")[0].strip() if ":" in clause else f"Clause {i}"
    } for i, clause in enumerate(eval_clauses)]
    db.add(embeddings, eval_clauses, metas)

    # Filter test queries corresponding to the 30 clauses
    subset_queries = [q for q in TEST_QUERIES if q["expected_idx"] < 30][:20]
    assert len(subset_queries) >= 15, "Insufficient test queries for CI evaluation gate."

    res = evaluate_query_list(db, subset_queries, w_bm25=0.10)
    
    assert res["r3"] >= 0.85, f"Recall@3 regression detected: {res['r3']:.3f} < 0.85"
    assert res["mrr"] >= 0.80, f"MRR regression detected: {res['mrr']:.3f} < 0.80"

def test_citation_hallucination_gate():
    """
    CI Gate: Verifies that post-hoc citation verification reliably catches fabricated citations.
    """
    real_chunks = [{
        "text": "The monthly rent is Rs. 22,000 payable on the 5th.",
        "meta": {"clause": "Monthly Rent", "source": "lease.txt"}
    }]

    # 1. Valid citation
    valid_answer = "The rent is Rs. 22,000 [Monthly Rent]"
    status, details = summarizer.verify_citations(valid_answer, real_chunks)
    assert status in ["supported", "partially supported"]

    # 2. Fabricated citation (hallucinated clause)
    fake_answer = "The rent is Rs. 22,000 [InventedSection_999]"
    status, details = summarizer.verify_citations(fake_answer, real_chunks)
    assert status == "citation error" or any(d.get("status") == "citation error" for d in details)

def test_upload_size_limit_gate():
    """
    CI Gate: Verifies that uploading files exceeding 15MB returns HTTP 413.
    """
    client = TestClient(app)
    # Generate 16 MB dummy content
    oversized_content = b"0" * (16 * 1024 * 1024)
    file_obj = BytesIO(oversized_content)

    response = client.post(
        "/upload",
        files={"file": ("large_file.txt", file_obj, "text/plain")}
    )
    assert response.status_code == 413
    assert "File size exceeds" in response.json()["detail"]
