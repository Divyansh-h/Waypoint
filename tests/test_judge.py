from unittest.mock import MagicMock, patch

from eval.judge import score_answer


@patch("eval.judge.genai.GenerativeModel")
def test_score_answer(mock_genai_model):
    # Setup mock
    mock_model_instance = MagicMock()
    mock_genai_model.return_value = mock_model_instance
    
    mock_response = MagicMock()
    mock_response.text = '{"is_correct": true, "no_hallucination": true, "is_complete": false, "multi_hop_synthesis": false, "has_citation": true}'
    mock_model_instance.generate_content.return_value = mock_response
    
    # Run evaluation
    result = score_answer(
        question="How do I fit a random forest?",
        answer="You use the fit() method.",
        retrieved_chunks=[{"content": "stuff", "file_path": "file"}]
    )
    
    assert result.total_score == 3
    assert result.is_correct is True
    mock_model_instance.generate_content.assert_called_once()
