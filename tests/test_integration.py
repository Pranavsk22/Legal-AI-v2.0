import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import os
from unittest.mock import patch, MagicMock

# Import the FastAPI app
from backend.api.main import app

def test_upload_and_build_index_flow(tmp_path):
    # Create a test text file to upload
    contract_file = tmp_path / "test_agreement.txt"
    contract_file.write_text(
        "Governed by Delaware law. Either party may terminate this agreement upon 30 days notice. The liability limit is $10,000.",
        encoding="utf-8"
    )
    
    client = TestClient(app)
    
    # Path patch targets
    test_index_path = tmp_path / "test_contracts.faiss"
    test_meta_path = tmp_path / "test_contracts_text.pkl"
    
    # Mock paths in routes and mock the requests.post for Groq completions
    with patch("backend.api.routes.INDEX_PATH", test_index_path), \
         patch("backend.api.routes.META_PATH", test_meta_path), \
         patch("backend.nlp_modules.summarizer.requests.post") as mock_post, \
         patch.dict("os.environ", {"GROQ_API_KEY_SUMMARY": "test-key", "GROQ_API_KEY_QA": "test-key"}):
         
        # Mock LLM API Response
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "This is a mock response from the LLM."
                    }
                }
            ]
        }
        mock_post.return_value = mock_resp
        
        # 1. Test POST /upload
        with open(contract_file, "rb") as f:
            response = client.post(
                "/upload",
                files={"file": ("test_agreement.txt", f, "text/plain")}
            )
            
        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "test_agreement.txt"
        assert data["chunks_added"] > 0
        assert data["summary"] == "This is a mock response from the LLM."
        
        # Verify index files built on disk correctly
        assert test_index_path.exists()
        assert test_meta_path.exists()
        
        # 2. Test POST /ask to query the generated index
        qa_response = client.post(
            "/ask",
            json={"query": "Which law governs?"}
        )
        
        assert qa_response.status_code == 200
        qa_data = qa_response.json()
        assert qa_data["answer"] == "This is a mock response from the LLM."
        assert len(qa_data["citations"]) > 0
        assert qa_data["citations"][0]["source"] == "test_agreement.txt"
        assert "Delaware" in qa_data["citations"][0]["snippet"]
