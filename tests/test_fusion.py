import pytest

from retrieval.fusion import reciprocal_rank_fusion


def test_rrf_math_correctness():
    """
    Verifies the underlying math of the Reciprocal Rank Fusion algorithm using
    a hand-computed example.
    """
    dense_results = [
        {"id": "docA", "content": "A"},
        {"id": "docB", "content": "B"},
        {"id": "docC", "content": "C"}
    ]
    
    bm25_results = [
        {"id": "docB", "content": "B"},
        {"id": "docA", "content": "A"},
        {"id": "docD", "content": "D"}
    ]
    
    # Using k = 2 for easier hand-computation:
    # docA: dense rank 1 -> 1/(2+1) = 1/3. bm25 rank 2 -> 1/(2+2) = 1/4. Total = 7/12 = 0.58333...
    # docB: dense rank 2 -> 1/(2+2) = 1/4. bm25 rank 1 -> 1/(2+1) = 1/3. Total = 7/12 = 0.58333...
    # docC: dense rank 3 -> 1/(2+3) = 1/5. bm25 none -> 0. Total = 1/5 = 0.2
    # docD: dense none -> 0. bm25 rank 3 -> 1/(2+3) = 1/5. Total = 1/5 = 0.2
    
    results = reciprocal_rank_fusion(dense_results, bm25_results, k=2, top_n=4)
    
    # Convert results list to a dictionary mapping id to rrf_score for easy assertion
    scores = {res["id"]: res["rrf_score"] for res in results}
    
    assert len(results) == 4
    
    assert scores["docA"] == pytest.approx(7/12)
    assert scores["docB"] == pytest.approx(7/12)
    assert scores["docC"] == pytest.approx(1/5)
    assert scores["docD"] == pytest.approx(1/5)
    
    # The first two elements should be A and B (order might vary due to python stable sorting on ties, 
    # but both should be the top 2)
    top_2_ids = {results[0]["id"], results[1]["id"]}
    assert top_2_ids == {"docA", "docB"}

def test_rrf_top_n_truncation():
    """
    Verifies that the RRF algorithm properly respects the top_n parameter.
    """
    dense = [{"id": f"doc{i}", "content": "dense"} for i in range(10)]
    bm25 = [{"id": f"doc{i}", "content": "bm25"} for i in range(9, -1, -1)]
    
    results = reciprocal_rank_fusion(dense, bm25, k=60, top_n=5)
    assert len(results) == 5
