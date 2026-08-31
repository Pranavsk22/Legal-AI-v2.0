# tests/test_draft_confirm_flow.py
import pytest
from fastapi.testclient import TestClient
from pathlib import Path
from unittest.mock import patch, MagicMock
from backend.api.main import app

def test_draft_confirm_flow(tmp_path):
    client = TestClient(app)
    
    # 1. Prepare temp file to upload to /ingest/draft
    draft_file = tmp_path / "draft_agreement.txt"
    draft_file.write_text(
        "Governed by California law. Parties: Alice and Bob. Monthly rent is $1500.",
        encoding="utf-8"
    )
    
    test_index_path = tmp_path / "test_confirm.faiss"
    test_meta_path = tmp_path / "test_confirm_text.pkl"
    
    with patch("backend.api.routes.INDEX_PATH", test_index_path), \
         patch("backend.api.routes.META_PATH", test_meta_path), \
         patch("backend.nlp_modules.summarizer.requests.post") as mock_post, \
         patch.dict("os.environ", {"GROQ_API_KEY_SUMMARY": "test-key", "GROQ_API_KEY_QA": "test-key"}):
         
        # Mock Groq metadata extraction response
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"doc_type": "Lease Agreement", "parties": ["Alice", "Bob"], "effective_date": "2026-08-28", "governing_law": "California"}'
                    }
                }
            ]
        }
        mock_post.return_value = mock_resp
        
        # Call /ingest/draft
        with open(draft_file, "rb") as f:
            response = client.post(
                "/ingest/draft",
                files={"file": ("draft_agreement.txt", f, "text/plain")}
            )
            
        assert response.status_code == 200
        data = response.json()
        assert data["doc_id"] == "draft_agreement.txt"
        assert data["doc_type"] == "Lease Agreement"
        assert "Alice" in data["parties"]
        assert len(data["chunks"]) > 0
        
        # Verify index was not built yet
        assert not test_index_path.exists()
        
        # Human edits the draft details and submits to /ingest/confirm
        confirm_payload = {
            "doc_id": data["doc_id"],
            "doc_type": "NDA", # Edited by human
            "risk_flags": data["risk_flags"],
            "parties": ["Alice", "Bob", "Charlie"], # Edited by human
            "effective_date": "2026-09-01", # Edited by human
            "governing_law": "Delaware", # Edited by human
            "source_format": data["source_format"],
            "chunks": data["chunks"]
        }
        
        # Call /ingest/confirm
        confirm_response = client.post(
            "/ingest/confirm",
            json=confirm_payload
        )
        
        assert confirm_response.status_code == 200
        confirm_data = confirm_response.json()
        assert confirm_data["status"] == "success"
        assert confirm_data["doc_id"] == "draft_agreement.txt"
        
        # Verify index is now created
        assert test_index_path.exists()
        assert test_meta_path.exists()
        
        # Search the confirmed doc to ensure it has edited fields
        search_response = client.get(
            "/search",
            params={"doc_type": "NDA", "governing_law": "Delaware"}
        )
        assert search_response.status_code == 200
        search_data = search_response.json()
        assert len(search_data) > 0
        assert search_data[0]["meta"]["doc_type"] == "NDA"
        assert "Charlie" in search_data[0]["meta"]["parties"]
