import pytest
from unittest.mock import patch, MagicMock
from ingestion.embed import embed_chunks
from ingestion.models import Chunk

@pytest.fixture
def sample_chunk_objects():
    return [
        Chunk(
            chunk_id="chunk1",
            name="func1",
            type="function",
            file_path="file1.py",
            content="def func1(): pass"
        ),
        Chunk(
            chunk_id="chunk2",
            name="func2",
            type="function",
            file_path="file2.py",
            content="def func2(): pass"
        )
    ]

@patch("ingestion.embed.requests.post")
@patch("ingestion.embed.os.environ.get")
def test_embed_chunks(mock_env_get, mock_post, sample_chunk_objects):
    mock_env_get.return_value = "fake_api_key"
    
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "data": [
            {"index": 0, "embedding": [0.1, 0.2]},
            {"index": 1, "embedding": [0.3, 0.4]}
        ]
    }
    mock_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_response

    embedded = embed_chunks(sample_chunk_objects, batch_size=2)
    
    assert len(embedded) == 2
    assert embedded[0].chunk_id == "chunk1"
    assert embedded[0].vector == [0.1, 0.2]
    assert embedded[1].chunk_id == "chunk2"
    assert embedded[1].vector == [0.3, 0.4]
    
    mock_post.assert_called_once()
