from typing import List, Set


def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Computes Recall at K.
    
    Recall@K measures what fraction of all perfectly relevant documents were 
    successfully retrieved within the top K results.
    
    Formula:
        Recall@K = |Retrieved_K ∩ Relevant| / |Relevant|
    
    Args:
        retrieved: List of retrieved document IDs, ordered by rank.
        relevant: Set of ground-truth relevant document IDs.
        k: The cutoff depth for retrieved documents.
        
    Returns:
        float: Recall score between 0.0 and 1.0. Returns 0.0 if relevant set is empty.
    """
    if not relevant:
        return 0.0
    
    retrieved_k = set(retrieved[:k])
    intersection = retrieved_k.intersection(relevant)
    return len(intersection) / len(relevant)


def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Computes Precision at K.
    
    Precision@K measures what fraction of the top K retrieved documents 
    were actually relevant.
    
    Formula:
        Precision@K = |Retrieved_K ∩ Relevant| / K
    
    Args:
        retrieved: List of retrieved document IDs, ordered by rank.
        relevant: Set of ground-truth relevant document IDs.
        k: The cutoff depth for retrieved documents.
        
    Returns:
        float: Precision score between 0.0 and 1.0. Returns 0.0 if k=0.
    """
    if k == 0:
        return 0.0
        
    retrieved_k = set(retrieved[:k])
    intersection = retrieved_k.intersection(relevant)
    return len(intersection) / k


def mrr(retrieved: List[str], relevant: Set[str]) -> float:
    """
    Computes the Reciprocal Rank (RR) for a single query.
    (Mean Reciprocal Rank is the average of this score across all queries).
    
    MRR evaluates how far down the list the *first* relevant document appears.
    
    Formula:
        RR = 1 / rank_i  (where rank_i is the 1-indexed position of the FIRST relevant doc)
        If no relevant doc is retrieved, RR = 0.0
    
    Args:
        retrieved: List of retrieved document IDs, ordered by rank.
        relevant: Set of ground-truth relevant document IDs.
        
    Returns:
        float: Reciprocal rank between 0.0 and 1.0.
    """
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            return 1.0 / (i + 1)
    
    return 0.0


def cluster_failures(misses: List[dict]) -> tuple[dict, dict]:
    """
    Groups missed questions by difficulty and question type to identify systemic weaknesses.
    
    Args:
        misses: List of evaluation result dictionaries that failed (hit=False).
        
    Returns:
        tuple: (difficulty_counts, type_counts) dictionaries
    """
    diff_counts = {}
    type_counts = {}
    
    for m in misses:
        diff = m.get("difficulty", "UNKNOWN").upper()
        qtype = m.get("type", "UNKNOWN").upper()
        diff_counts[diff] = diff_counts.get(diff, 0) + 1
        type_counts[qtype] = type_counts.get(qtype, 0) + 1
        
    return diff_counts, type_counts
