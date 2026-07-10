import logging
import os
import time
import concurrent.futures
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
    Transforms raw text Chunks into EmbeddedChunks by orchestrating API calls.
    This function uses a ThreadPoolExecutor to run batches concurrently, which is critical
    because sequential network IO for thousands of chunks would otherwise become a massive,
    blocking bottleneck in the ingestion pipeline. It deliberately re-sorts results to maintain
    the original file order for deterministic indexing.
    """
    if not os.environ.get("JINA_API_KEY"):
        logger.warning("JINA_API_KEY environment variable is not set. Embedding will likely fail.")
        
    embedded_chunks = []
    
    # Create batches
    batches = [chunks[i : i + batch_size] for i in range(0, len(chunks), batch_size)]
    
    def process_batch(batch_idx, batch):
        texts = [chunk.content for chunk in batch]
        embeddings = get_jina_embeddings(texts)
        
        if len(embeddings) != len(batch):
            raise ValueError(f"Jina API returned {len(embeddings)} embeddings for {len(batch)} chunks.")
            
        result = []
        for chunk, emb in zip(batch, embeddings):
            result.append(
                EmbeddedChunk(
                    **chunk.model_dump(),
                    vector=emb
                )
            )
        return batch_idx, result

    # Execute in parallel with 5 workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(process_batch, i, b): i for i, b in enumerate(batches)}
        
        # We need to re-sort results based on batch_idx to maintain file/line order, 
        # as as_completed yields out of order
        results_by_idx = {}
        for future in concurrent.futures.as_completed(futures):
            try:
                batch_idx, batch_result = future.result()
                results_by_idx[batch_idx] = batch_result
                logger.info(f"Embedded batch {batch_idx+1}/{len(batches)}.")
            except Exception as e:
                logger.error(f"Batch embedding failed: {e}")
                
    # Reassemble in exact original order
    for i in range(len(batches)):
        if i in results_by_idx:
            embedded_chunks.extend(results_by_idx[i])
            
    return embedded_chunks
