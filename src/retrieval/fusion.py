from typing import Any, Dict, List


def reciprocal_rank_fusion(
    dense_results: List[Dict[str, Any]], 
    bm25_results: List[Dict[str, Any]], 
    k: int = 60,
    top_n: int = 10
) -> List[Dict[str, Any]]:
    """
    Combines two ranked lists using Reciprocal Rank Fusion (RRF).
    RRF is robust to score scale mismatches because it relies strictly on ranks.
    
    Formula: RRF_score = 1 / (k + rank)
    
    Args:
        dense_results: Ranked list of chunks from dense retrieval (assumed sorted).
        bm25_results: Ranked list of chunks from BM25 retrieval (assumed sorted).
        k: The smoothing constant for RRF (default: 60).
        top_n: Number of final fused results to return.
        
    Returns:
        List of fused chunk dictionaries sorted by RRF score descending.
    """
    fused_scores: Dict[str, float] = {}
    combined_docs: Dict[str, Dict[str, Any]] = {}
    
    for rank, doc in enumerate(dense_results, 1):
        doc_id = doc["id"]
        fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        combined_docs[doc_id] = doc
        
    for rank, doc in enumerate(bm25_results, 1):
        doc_id = doc["id"]
        fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        if doc_id not in combined_docs:
            combined_docs[doc_id] = doc
            
    sorted_items = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    
    fused_results = []
    for doc_id, score in sorted_items[:top_n]:
        fused_doc = combined_docs[doc_id].copy()
        fused_doc["rrf_score"] = score
        fused_results.append(fused_doc)
        
    return fused_results

