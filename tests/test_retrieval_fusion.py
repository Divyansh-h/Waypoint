import sys
from pathlib import Path
import pytest

# Ensure src/ is in the python path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir / "src"))

from retrieval.fusion import reciprocal_rank_fusion, weighted_score_fusion

def test_reciprocal_rank_fusion_consensus():
    """
    Test that RRF correctly bubbles up consensus items.
    'chunk_C' is ranked #1 in both lists, so it should win easily.
    """
    dense_results = [
        {"id": "chunk_C", "score": 0.9}, # Rank 1
        {"id": "chunk_A", "score": 0.8}, # Rank 2
        {"id": "chunk_B", "score": 0.7}  # Rank 3
    ]
    
    bm25_results = [
        {"id": "chunk_C", "score": 25.0}, # Rank 1
        {"id": "chunk_B", "score": 15.0}, # Rank 2
        {"id": "chunk_A", "score": 10.0}  # Rank 3
    ]
    
    fused = reciprocal_rank_fusion(dense_results, bm25_results, k=60, top_n=3)
    
    # --- STUB ASSERTIONS ---
    # Uncomment these once you implement reciprocal_rank_fusion in src/retrieval/fusion.py
    # assert len(fused) == 3
    # assert fused[0]["id"] == "chunk_C", "Consensus #1 should win."
    # assert fused[1]["id"] == "chunk_A", "RRF should resolve the tie correctly based on the formula."
    pass


def test_weighted_score_fusion_normalization():
    """
    Test that Simple Weighted Score correctly min-max normalizes scores
    before applying the weights. Without normalization, chunk_B would
    dominate purely because of its massive raw BM25 score.
    """
    dense_results = [
        {"id": "chunk_A", "score": 0.9},
        {"id": "chunk_B", "score": 0.5}
    ]
    
    bm25_results = [
        {"id": "chunk_B", "score": 4500.0}, # Massive outlier score
        {"id": "chunk_A", "score": 2.0}
    ]
    
    fused = weighted_score_fusion(dense_results, bm25_results, dense_weight=0.7, bm25_weight=0.3, top_n=2)
    
    # --- STUB ASSERTIONS ---
    # Uncomment these once you implement weighted_score_fusion with normalization
    # assert len(fused) == 2
    pass
