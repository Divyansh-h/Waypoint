import json
import logging
from typing import Dict, Set

from sentence_transformers.evaluation import InformationRetrievalEvaluator

logger = logging.getLogger(__name__)

def create_ir_evaluator(
    val_dataset_path: str, name: str = "val_eval"
) -> InformationRetrievalEvaluator:
    """
    Parses the validation JSONL dataset and constructs an InformationRetrievalEvaluator.
    This evaluator calculates strict retrieval metrics (Recall@k, MRR@k, NDCG@k) 
    during the HuggingFace Training loop at the end of each epoch.
    
    Expected JSONL format: 
    {"anchor": "query text", "positive": "target code chunk", "chunk_id": "optional_id"}
    """
    queries: Dict[str, str] = {}
    corpus: Dict[str, str] = {}
    relevant_docs: Dict[str, Set[str]] = {}
    
    logger.info(f"Loading validation dataset for evaluation from: {val_dataset_path}")
    
    try:
        with open(val_dataset_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                data = json.loads(line)
                
                # Extract fields
                query = data.get("anchor")
                chunk = data.get("positive")
                
                if not query or not chunk:
                    continue
                    
                # Create unique IDs for the evaluation mappings
                query_id = f"q_{idx}"
                doc_id = data.get("chunk_id", f"d_{idx}")
                
                # Populate mappings required by InformationRetrievalEvaluator
                queries[query_id] = query
                corpus[doc_id] = chunk
                relevant_docs[query_id] = {doc_id}
                
    except FileNotFoundError:
        logger.error(f"Validation dataset not found at {val_dataset_path}")
        raise
        
    logger.info(f"Loaded {len(queries)} evaluation queries and {len(corpus)} corpus documents.")
            
    # Initialize the SentenceTransformers standard IR Evaluator
    evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        show_progress_bar=True,
        name=name,
        # Track the metrics that exactly mirror our Phase 1 telemetry
        mrr_at_k=[10],
        ndcg_at_k=[10],
        accuracy_at_k=[1, 5, 10],
        precision_recall_at_k=[5, 10],
        map_at_k=[10]
    )
    
    return evaluator
