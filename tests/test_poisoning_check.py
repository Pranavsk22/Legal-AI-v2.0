# tests/test_poisoning_check.py
import pytest
import numpy as np
from backend.nlp_modules.vector_store import VectorDB

def test_detect_poisoning_no_warnings():
    # 5 consistent vectors (all identical normalized vectors)
    embeddings = []
    for _ in range(5):
        vec = np.zeros(384)
        vec[0] = 1.0
        embeddings.append(vec)
        
    db = VectorDB(dim=384)
    warnings = db.detect_poisoning(embeddings)
    
    assert len(warnings) == 0

def test_detect_poisoning_with_outlier():
    # 4 consistent vectors, 1 outlier vector that is completely orthogonal
    embeddings = []
    for _ in range(4):
        vec = np.zeros(384)
        vec[0] = 1.0
        embeddings.append(vec)
        
    # Outlier vector
    outlier = np.zeros(384)
    outlier[1] = 1.0 # orthogonal to the rest (similarity will be 0.0)
    embeddings.append(outlier)
    
    db = VectorDB(dim=384)
    warnings = db.detect_poisoning(embeddings)
    
    assert len(warnings) == 1
    assert "Chunk 4" in warnings[0]
    assert "anomalously dissimilar" in warnings[0]

def test_detect_poisoning_too_short():
    # Document with only 2 chunks shouldn't run outlier check
    embeddings = [np.ones(384), np.ones(384)]
    db = VectorDB(dim=384)
    warnings = db.detect_poisoning(embeddings)
    assert len(warnings) == 0
