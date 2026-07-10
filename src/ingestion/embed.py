import os
import time
import requests
import logging
from typing import List

from .models import Chunk, EmbeddedChunk

logger = logging.getLogger("ingestion_pipeline")

JINA_API_URL = "https://api.jina.ai/v1/embeddings"
JINA_API_KEY = os.environ.get("JINA_API_KEY", "")

def get_jina_embeddings(texts: List[str], retries: int = 3) -> List[List[float]]:
    """Calls the Jina API to get embeddings with exponential backoff retry logic."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {JINA_API_KEY}"
    }
    data = {
        "model": "jina-embeddings-v3",
        "task": "retrieval.passage", # Jina v3 specific task instruction
        "input": texts
    }
    
    for attempt in range(retries):
        try:
            response = requests.post(JINA_API_URL, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            # Ensure the embeddings match the input order
            sorted_data = sorted(result["data"], key=lambda x: x["index"])
            return [item["embedding"] for item in sorted_data]
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Jina API request failed (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                sleep_time = 2 ** attempt
                logger.info(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                logger.error("Max retries reached for Jina API.")
                raise e
    return []

def embed_chunks(chunks: List[Chunk], batch_size: int = 64) -> List[EmbeddedChunk]:
    """
    Takes a list of chunks, batches them, and queries the Jina embeddings API.
    """
    if not JINA_API_KEY:
        logger.warning("JINA_API_KEY environment variable is not set. Embedding will likely fail.")
        
    embedded_chunks = []
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [chunk.content for chunk in batch]
        
        batch_num = (i // batch_size) + 1
        logger.info(f"Embedding batch {batch_num}/{total_batches} ({len(batch)} chunks)...")
        
        embeddings = get_jina_embeddings(texts)
        
        for chunk, vector in zip(batch, embeddings):
            embedded_chunks.append(EmbeddedChunk(
                content=chunk.content,
                file_path=chunk.file_path,
                chunk_type=chunk.chunk_type,
                function_name=chunk.function_name,
                section_path=chunk.section_path,
                line_start=chunk.line_start,
                line_end=chunk.line_end,
                vector=vector
            ))
            
    return embedded_chunks
