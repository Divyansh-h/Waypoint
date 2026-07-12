import os
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class Reranker:
    """
    A cross-encoder reranker that evaluates query-document pairs simultaneously 
    to provide high-precision relevance scores.
    """
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", cache_dir: str = ".models_cache"):
        self.model_name = model_name
        self.cache_dir = cache_dir
        
        # Ensure HF and SentenceTransformers use our local project cache
        os.environ["HF_HOME"] = os.path.abspath(self.cache_dir)
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = os.path.abspath(self.cache_dir)
        
        if self.model_name != "stub":
            try:
                from sentence_transformers import CrossEncoder
                logger.info(f"Loading CrossEncoder model: {self.model_name}...")
                self.model = CrossEncoder(self.model_name)
            except ImportError:
                logger.error("sentence-transformers is not installed. Please install it.")
                self.model = None
        else:
            self.model = None
        
    def rerank(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Reranks a list of candidate documents based on their semantic relevance to the query.
        
        Args:
            query: The user's search query.
            candidates: A list of dictionaries representing the chunks. 
                        Expected to have at least an 'id' and 'content' field.
                        
        Returns:
            A new list of candidate dictionaries sorted by the reranker's score (descending).
        """
        if not candidates:
            return []
            
        if self.model_name == "stub" or self.model is None:
            # Fallback to returning in original order
            return candidates
            
        # Format input for the cross-encoder: a list of [query, document] pairs
        pairs = [[query, c["content"]] for c in candidates]
        
        # Predict relevance scores
        scores = self.model.predict(pairs)
        
        # Assign scores back to candidates
        for i, candidate in enumerate(candidates):
            candidate["reranker_score"] = float(scores[i])
            
        # Sort descending by the cross-encoder score
        return sorted(candidates, key=lambda x: x.get("reranker_score", 0), reverse=True)

_instance = None

def get_reranker(model_name: str, cache_dir: str = ".models_cache") -> Reranker:
    """
    Singleton getter to prevent reloading the heavy cross-encoder model into memory
    on every single query evaluation.
    """
    global _instance
    if _instance is None or _instance.model_name != model_name:
        _instance = Reranker(model_name, cache_dir)
    return _instance
