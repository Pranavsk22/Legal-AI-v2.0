import pytest
from unittest.mock import patch, MagicMock
from backend.nlp_modules.summarizer import summarize_document, answer_question, summarize_with_groq

@patch("backend.nlp_modules.summarizer.requests.post")
def test_summarize_document_mock(mock_post):
    # Setup mock API response
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "This is a mock legal summary."
                }
            }
        ]
    }
    mock_post.return_value = mock_resp
    
    # Run summarization (using mock keys environment variables)
    with patch.dict("os.environ", {"GROQ_API_KEY_SUMMARY": "test-key"}):
        summary = summarize_document("Test contract text.")
        
    assert summary == "This is a mock legal summary."
    mock_post.assert_called_once()

@patch("backend.nlp_modules.summarizer.requests.post")
def test_answer_question_mock(mock_post):
    # Setup mock API response
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "Mock Answer: Yes, liability is limited [Clause 10]."
                }
            }
        ]
    }
    mock_post.return_value = mock_resp
    
    # Run question answering
    with patch.dict("os.environ", {"GROQ_API_KEY_QA": "test-key"}):
        answer = answer_question("Is liability limited?", "Context showing Clause 10 limits liability.")
        
    assert "Mock Answer" in answer
    assert "[Clause 10]" in answer
    mock_post.assert_called_once()

@patch("backend.nlp_modules.summarizer.requests.post")
def test_summarize_with_groq_routing(mock_post):
    # Setup mock response
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "Mock routed response."
                }
            }
        ]
    }
    mock_post.return_value = mock_resp
    
    with patch.dict("os.environ", {"GROQ_API_KEY_SUMMARY": "test-key"}):
        # 1. Single parameter call -> summarize_document
        res1 = summarize_with_groq("Some context to summarize")
        assert res1 == "Mock routed response."
        
        # 2. Two parameter call -> answer_question
        res2 = summarize_with_groq("What is the term?", "This agreement is for one year.")
        assert res2 == "Mock routed response."
        
    assert mock_post.call_count == 2
