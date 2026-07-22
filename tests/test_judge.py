import pytest
from unittest.mock import patch, MagicMock
from eval.judge import validate_synthesis

@patch("eval.judge.genai.GenerativeModel")
def test_validate_synthesis(mock_genai_model):
    # Setup mock
    mock_model_instance = MagicMock()
    mock_genai_model.return_value = mock_model_instance
    
    mock_response = MagicMock()
    mock_response.text = '{"reasoning": "The response is correct and helpful.", "rating": 5}'
    mock_model_instance.generate_content.return_value = mock_response
    
    # Run evaluation
    result = validate_synthesis(
        user_query="How do I fit a random forest?",
        agent_response="You use the fit() method.",
        expected_behavior="Should mention fit method."
    )
    
    assert result.rating == 5
    assert "correct" in result.reasoning
    mock_model_instance.generate_content.assert_called_once()
