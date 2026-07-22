import logging
import os
import time
from typing import List

import requests

from .models import Chunk, EmbeddedChunk

logger = logging.getLogger("ingestion_pipeline")

JINA_API_URL = "https://api.jina.ai/v1/embeddings"


def get_jina_embeddings(texts: List[str], retries: int = 3) -> List[List[float]]:
    """
    Interfaces directly with the Jina embeddings API to convert code/text strings into dense vectors.
    This function abstracts away the network layer and intentionally implements exponential backoff 
    because batch processing thousands of chunks will inevitably hit transient network failures 
    or API rate limits, which would otherwise crash the entire pipeline mid-run.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
        
    api_key = os.environ.get("JINA_API_KEY", "")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
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
    Transforms raw text Chunks into EmbeddedChunks by orchestrating batched API calls.
    Processes batches sequentially with a 1-second delay between batches to respect
    free-tier API rate limits (429 avoidance).
    """
    if not os.environ.get("JINA_API_KEY"):
        logger.warning("JINA_API_KEY environment variable is not set. Embedding will likely fail.")
        
    embedded_chunks: List[EmbeddedChunk] = []
    
    # Create batches
    batches = [chunks[i : i + batch_size] for i in range(0, len(chunks), batch_size)]
    
    for batch_idx, batch in enumerate(batches):
        texts = [chunk.content for chunk in batch]
        
        try:
            embeddings = get_jina_embeddings(texts)
        
            if len(embeddings) != len(batch):
                raise ValueError(
                    f"Jina API returned {len(embeddings)} embeddings for {len(batch)} chunks."
                )
            
            for chunk, emb in zip(batch, embeddings):
                embedded_chunks.append(
                    EmbeddedChunk(
                        **chunk.model_dump(),
                        vector=emb
                    )
                )
            logger.info(f"Embedded batch {batch_idx + 1}/{len(batches)}.")
        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} embedding failed: {e}")
        
        # Rate limit protection between batches
        if batch_idx < len(batches) - 1:
            time.sleep(1)
            
    return embedded_chunks

