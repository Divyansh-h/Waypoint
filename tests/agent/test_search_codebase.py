import pytest
from unittest.mock import patch, MagicMock
from agent.tools.search_codebase import SearchCodebaseTool

def test_search_codebase_execute():
    # Mocking both psycopg2.connect and the RetrievalPipeline
    with patch("psycopg2.connect") as mock_connect, \
         patch("agent.tools.search_codebase.RetrievalPipeline") as mock_pipeline_cls:
        
        # Setup mocks
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        
        mock_pipeline = MagicMock()
        mock_pipeline_cls.return_value = mock_pipeline
        
        # Instantiate tool
        tool = SearchCodebaseTool()
        
        # Test missing query
        assert "ERROR: Missing 'query'" in tool.execute({})
        
        # Test empty retrieval
        mock_pipeline.retrieve.return_value = []
        assert "No semantic overlap found" in tool.execute({"query": "test query"})
        
        # Test successful retrieval
        mock_pipeline.retrieve.return_value = ["chunk_1"]
        
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        # Mock fetchall to return one row
        mock_cursor.fetchall.return_value = [("chunk_1", "def test_func():\n    pass")]
        
        result = tool.execute({"query": "test query"})
        assert "SNIPPET ID: chunk_1" in result
        assert "def test_func():" in result
