import pytest
from unittest.mock import patch, MagicMock
from ingestion.indexer import PgVectorIndexer
from ingestion.models import EmbeddedChunk

@patch("ingestion.indexer.psycopg2.connect")
def test_indexer_upsert(mock_connect):
    mock_conn = MagicMock()
    mock_connect.return_value = mock_conn
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    
    indexer = PgVectorIndexer()
    
    chunks = [
        EmbeddedChunk(
            chunk_id="chunk1",
            name="func1",
            type="function",
            file_path="file1.py",
            content="def func1(): pass",
            vector=[0.1]*1024
        )
    ]
    
    with patch("ingestion.indexer.execute_batch") as mock_execute_batch:
        indexer.index_chunks(chunks)
        mock_execute_batch.assert_called_once()
        args, kwargs = mock_execute_batch.call_args
        assert args[0] == mock_cursor
        assert len(kwargs["argslist"]) == 1
        assert kwargs["argslist"][0][0] == "chunk1"
