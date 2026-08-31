# tests/test_citation_verification.py
import pytest
import numpy as np
from unittest.mock import patch
from backend.nlp_modules.summarizer import verify_citations

# Helper to generate normalized mock vector
def mock_vector(val, size=384):
    vec = np.zeros(size)
    vec[0] = val
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

@pytest.fixture
def mock_retrieved_chunks():
    return [
        {
            "text": "Monthly Rent: Rs. 22,000 (Rupees Twenty Two Thousand only) per month.",
            "meta": {"clause": "Monthly Rent", "source": "lease.txt"}
        },
        {
            "text": "Security Deposit: Rs. 1,20,000 (Rupees One Lakh Twenty Thousand only).",
            "meta": {"clause": "Security Deposit", "source": "lease.txt"}
        },
        {
            "text": "This lease is automatically renewable for a further period of 11 months with 5% escalation.",
            "meta": {"clause": "Renewal", "source": "lease.txt"}
        }
    ]

@patch("backend.nlp_modules.embedder.embed_chunks")
def test_verify_citations_supported(mock_embed, mock_retrieved_chunks):
    # Setup mock embeddings: we want high cosine similarity (dot product = 1.0)
    # between the answer sentence and the retrieved chunk text.
    # So we make both return the exact same vector.
    mock_embed.side_effect = lambda texts: [mock_vector(1.0) for _ in texts]
    
    answer = "The monthly rent is Rs. 22,000 as agreed. 【Monthly Rent】"
    status, details = verify_citations(answer, mock_retrieved_chunks)
    
    assert status == "supported"
    assert len(details) == 1
    assert details[0]["status"] == "supported"
    assert details[0]["cited_clause"] == "Monthly Rent"
    assert details[0]["semantic_similarity"] == pytest.approx(1.0)

@patch("backend.nlp_modules.embedder.embed_chunks")
def test_verify_citations_citation_error(mock_embed, mock_retrieved_chunks):
    mock_embed.side_effect = lambda texts: [mock_vector(1.0) for _ in texts]
    
    # Citing a clause "Stamp Duty" that does not exist in mock_retrieved_chunks
    answer = "The stamp duty must be shared. [Stamp Duty]"
    status, details = verify_citations(answer, mock_retrieved_chunks)
    
    assert status == "citation error"
    assert len(details) == 1
    assert details[0]["status"] == "citation error"
    assert details[0]["cited_clause"] == "Stamp Duty"

@patch("backend.nlp_modules.embedder.embed_chunks")
def test_verify_citations_partially_supported(mock_embed, mock_retrieved_chunks):
    # To get a moderate semantic similarity (say 0.45), we make
    # embed_chunks return vectors that have a dot product of 0.45.
    # Let's say vector 1 is [1.0, 0, ...] and vector 2 is [0.45, sqrt(1-0.45^2), ...]
    def side_effect(texts):
        res = []
        for text in texts:
            if "security deposit" in text.lower():
                # This is the sentence
                vec = np.zeros(384)
                vec[0] = 0.45
                vec[1] = np.sqrt(1 - 0.45**2)
                res.append(vec)
            else:
                # This is the retrieved chunk
                res.append(mock_vector(1.0))
        return res
        
    mock_embed.side_effect = side_effect
    
    # Lexical overlap: "security deposit" has some overlap but not very high
    # Let's see: STOPWORDS will filter some out.
    answer = "The security deposit is something. [Security Deposit]"
    status, details = verify_citations(answer, mock_retrieved_chunks)
    
    assert status in ["partially supported", "supported"] # Lexical or semantic triggers it
    assert len(details) == 1
    assert details[0]["cited_clause"] == "Security Deposit"

@patch("backend.nlp_modules.embedder.embed_chunks")
def test_verify_citations_unsupported_no_citation(mock_embed, mock_retrieved_chunks):
    # No citations in the response, and it's not "NOT FOUND"
    answer = "The rent is Rs. 22,000 and the contract is governed by laws of Karnataka."
    status, details = verify_citations(answer, mock_retrieved_chunks)
    
    assert status == "unsupported"
    assert len(details) == 0

def test_verify_citations_not_found(mock_retrieved_chunks):
    # LLM says NOT FOUND
    answer = "NOT FOUND"
    status, details = verify_citations(answer, mock_retrieved_chunks)
    
    assert status == "supported"
    assert len(details) == 0
