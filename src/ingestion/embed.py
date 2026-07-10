from typing import List
from .models import Chunk, EmbeddedChunk
# import openai  # To be used later when implementing the stub

def embed_chunks(chunks: List[Chunk]) -> List[EmbeddedChunk]:
    """
    Takes a list of AST chunks and queries the embedding API (e.g., OpenAI's text-embedding-3-small)
    to generate vector embeddings for each chunk.

    Args:
        chunks: A list of Chunk models containing raw text and metadata.

    Returns:
        A list of EmbeddedChunk models containing the original data plus the vector.
    """
    embedded_chunks = []
    
    # TODO: Implement batching logic to send chunks to the embedding model
    # (e.g., OpenAI batch API) to stay within rate limits and optimize speed.
    for chunk in chunks:
        # STUB: Replace this with the actual API call
        dummy_vector = [0.0] * 1536  # text-embedding-3-small outputs 1536 dims by default
        
        embedded_chunk = EmbeddedChunk(
            content=chunk.content,
            file_path=chunk.file_path,
            chunk_type=chunk.chunk_type,
            function_name=chunk.function_name,
            line_start=chunk.line_start,
            line_end=chunk.line_end,
            vector=dummy_vector
        )
        embedded_chunks.append(embedded_chunk)
        
    return embedded_chunks
